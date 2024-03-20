"""This is used to train the Question answering model based on new provided data.

Support logic:
- Load data file
- Do prepprocessing to get dataset
- Split to train and validation data
- Fine-tuning: 1. full parameters 2. peft training
    - Load pretrained model and tokenizer that supports for the QA task
    - FEPT model with LORA based model 
- Save pretrained model checkpoint, and evalute it, dump metrics to disk for comparation
- Get prediction for QA pairs

-> Use the predicted answer to construct the NEXT LLM model to analysis the risk for clause.
"""
import json
import pandas as pd
from datasets import Dataset, load_from_disk
import torch
import shutil
import os
import numpy as np

import argparse
import sys
from transformers import TrainingArguments
from transformers import Trainer
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from transformers import Trainer, TrainingArguments


qa_model_dict = {
    'dist_bert':"distilbert-base-cased-distilled-squad",
    'roberta': "deepset/roberta-base-squad2",
    'bert': "deepset/bert-base-cased-squad2",
    'bart': "valhalla/bart-large-finetuned-squadv1"
}

is_cuda = torch.cuda.is_available()

device = torch.device('cuda') if is_cuda else torch.device('cpu')

global model, tokenizer
max_length = 384
stride = 128
cur_path = os.path.abspath(os.curdir)


def get_model_tokenizer(model_id):
    model_id = qa_model_dict.get(model_id)
    if not model_id:
        print("Couldn't get the model that current support: {}".format(model_id))
        return 
    
    print("Start to Download pre-trained model")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForQuestionAnswering.from_pretrained(model_id)

    try:
        model.to(device)
    except Exception as e:
        print("When to put model to GPU with error: {}".format(e))
    return model, tokenizer


def load_json(file_name):
    with open(file_name, 'r') as f:
        data = json.loads(f.read())
    return data


# let's create a func to make the real data with 
def _get_trained_data(training_data_path='sample.json'):
    train_data = load_json(training_data_path)

    real_train_ds = []
    for i, d in enumerate(train_data['data']):
        ps = d['paragraphs']
        for p in ps:
            context = p['context']
            for qas in p['qas']:
                qas['context'] = context
                qas.pop('is_impossible', '')
                if qas['answers'] == 0:
                    continue
                if len(qas['answers'])  >= 1:
                    # print(qas['answers'])
                    tmp_ans = qas['answers'][0]
                    tmp_ans['text'] = [tmp_ans['text']]
                    tmp_ans['answer_start'] = [tmp_ans['answer_start']]
                    qas['answers'] = tmp_ans
                
                real_train_ds.append(qas) 

    real_train_ds = [x for x in real_train_ds if x['answers'] != []]
    return real_train_ds


def _get_test_data(data_path='test.json'):
    test_data = load_json(data_path)
    test_data_list = []
    for i, d in enumerate(test_data['data']):
        ps = d['paragraphs']
        for p in ps:
            context = p['context']
            for qas in p['qas']:
                qas['context'] = context
                qas.pop('is_impossible', '')
            test_data_list.append(qas)
    return test_data_list



def _get_dataset_from_json(json_data):
    df = pd.DataFrame(json_data)

    dataset = Dataset.from_pandas(df)
    print('get dataest size: ', len(dataset))
    return dataset


def preprocess_training_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


def preprocess_validation_examples(examples):
    if isinstance(examples, dict):
        questions = examples["question"].strip()
    else:
        questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs


def get_validation_dataset(validation_file_path=None, validation_json=None):
    """get validation dataset based on the file or a json object

    Args:
        validation_file_path (_type_, optional): _description_. Defaults to None.
        validation_json (JSON, optional): Must with `question`, `context`, `id`. Defaults to None.
    """
    if not validation_file_path:
        # load from local file
        validation_json = _get_test_data(data_path=validation_file_path)
        vali_ds = _get_dataset_from_json(validation_json)
    else:
        if not isinstance(validation_json, list):
            validation_json = [validation_json]
        vali_ds = _get_dataset_from_json(validation_json)
    return vali_ds


def get_model_prediction(model, question, context, id, n_best = 20, max_answer_length=300):
    example = {
        'question': question, 
        'context': context,
        'id': id
    }
    inputs = preprocess_validation_examples(example)
    
    if 'offset_mapping' in inputs:
        offsets = inputs.pop('offset_mapping')
    
    # batch = {k: inputs[k].to(device) for k in inputs}
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    start_logits = outputs.start_logits.cpu().numpy()
    end_logits = outputs.end_logits.cpu().numpy()
    
    predicted_answers = []

    example_id = example["id"]
    context = example["context"]
    answers = []

    # TODO: confirm whether this is only the 0 index?
    start_logit = start_logits[0]
    end_logit = end_logits[0]

    start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
    end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
    for start_index in start_indexes:
        for end_index in end_indexes:
            # Skip answers that are not fully in the context
            if offsets[start_index] is None or offsets[end_index] is None:
                continue
            # Skip answers with a length that is either < 0 or > max_answer_length.
            if (
                end_index < start_index
                or end_index - start_index + 1 > max_answer_length
            ):
                continue

            answers.append(
                {
                    "text": context[offsets[start_index][0] : offsets[end_index][1]],
                    "logit_score": start_logit[start_index] + end_logit[end_index],
                }
            )

    best_answer = max(answers, key=lambda x: x["logit_score"])
    predicted_answers.append({"id": example_id, "prediction_text": best_answer["text"]})
    
    return predicted_answers
        
    


def validate_model(model, test_data_path='test.json'):
    vali_ds = _get_test_data(test_data_path)
    validation_dataset = vali_ds.map(
        preprocess_validation_examples,
        batched=True,
        remove_columns=vali_ds.column_names,
    )

    eval_set_for_model = validation_dataset.remove_columns(["example_id", "offset_mapping"])
    eval_set_for_model.set_format("torch")

    batch = {k: eval_set_for_model[k].to(device) for k in eval_set_for_model.column_names}

    with torch.no_grad():
        outputs = model(**batch)
        
    
        

def get_train_dataset(json_path, data_path='tmp_data'):
    if os.path.exists(data_path):
        print("Start to load dataset from disk!")
        dataset = load_from_disk(data_path)
    else:
        print("Start to build dataset based on JSON file")
        train_json = _get_trained_data(json_path)
        dataset  = _get_dataset_from_json(train_json)
        dataset = dataset.map(preprocess_training_examples, batched=True, remove_columns=dataset.column_names)
         # split to train and test dataset
        dataset = dataset.train_test_split(test_size=.1)
        dataset.save_to_disk(data_path)
    return dataset


def _get_latest_checkpoint(folder_path):
    """Used to get the latest checkpoint to get the latest trained model.

    Args:
        folder_path (_type_): _description_

    Returns:
        _type_: _description_
    """
    check_point_folder_list = os.listdir(folder_path)
    if len(check_point_folder_list) == 0:
        print("No checkpoint folder get")
        return
    # get the lastest one
    latest_folder = sorted(check_point_folder_list, key=lambda x: x.split('-')[-1], reverse=True)

    return latest_folder[0]


def _dump_json_metric(model_name, info_dict, metric_path='metrics', ):
    # after the full process finsished, then we could loop this folder to get the training info, 
    # and sort the the metrics, then to get the best trained model, use this model to do prediction.
    metric_path = os.path.join(cur_path, metric_path)
    if not os.path.exists(metric_path):
        os.makedirs(metric_path, exist_ok=True)
    model_path = os.path.join(metric_path, model_name + '.json')
    with open(model_path, 'w') as f:
        print("Start to dump json to metric path: {}".format(model_path))
        f.write(json.dumps(info_dict))


def train_model(model, tokenizer, dataset,  output_dir=None, training_config={}):
    if not output_dir:
        output_dir = 'model_output'
    # batch_size 64 is tested will cause 10GB GPU memory
    batch_size = training_config.get('batch_size', 12)
    args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=1,
        weight_decay=0.01,
        per_device_train_batch_size=batch_size,
        fp16=True,
        push_to_hub=False,
        logging_steps=100,
        report_to='none'
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        tokenizer=tokenizer
    )
    training_info = trainer.train()

    evalution_info = trainer.evaluate()
    info_dic = {'training': training_info, 'evalute': evalution_info}
    
    # as some model_id will contain / to separate
    model_path = model_id.split('/')[-1] if '/' in model_id else model_id
    _dump_json_metric(model_name=model_id, info_dict=info_dic)
    
     
def get_peft_model_lora_based(model, config=None):
    if not config:
        from peft import get_peft_model, LoraConfig, TaskType
        
        config = LoraConfig(task_type=TaskType.QUESTION_ANS, inference_mode=False, r=8, lora_alpha=32, lora_dropout=.1, target_modules="all-linear")
    # before to get lora model,fix the full model parameters
    for p in model.parameters():
        p.reguires_grad = False
    model_new = get_peft_model(model, config)
    print("Get new trainable model parameters: ")
    model_new.print_trainable_parameters()
    return model_new


def _load_model_tokenizer(model_path):
    if not os.path.exists(model_path):
        print("No model path find: {}".format(model_path))
        return 
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer



def get_model_prediction(question, context, model, tokenizer):
    inputs = tokenizer(
        question,
        context,
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        padding="max_length",
        return_tensors='pt'
    )
    
    is_model_in_gpu = next(model.parameters()).is_cuda
    
    if is_model_in_gpu:
        inputs.to(device)
    
    with torch.no_grad():
        out = model(**inputs)
        
    answer_start_index = out.start_logits.argmax()
    answer_end_index = out.end_logits.argmax()
    
    predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
    predict_str = tokenizer.decode(predict_answer_tokens)
    return predict_str



if __name__ == '__main__':
    # to make the code to be called for shell, just add argparser, then will call different logic
    # get the predefined model list that support QA question.
    # before training, we should empty the cuda cache
    """qa_model_dict = {
        'dist_bert':"distilbert-base-cased-distilled-squad",
        'roberta': "deepset/roberta-base-squad2",
        'bert': "deepset/bert-base-cased-squad2",
        'bart': "valhalla/bart-large-finetuned-squadv1"
        }"""    
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_id', '-md', type=str, help='which model to use? Support list: [dist_bert, bert, bart, roberta]')
    parser.add_argument('--batch_size', type=int, default=12, help='batch_size to use')
    args = parser.parse_args()

    model_id = args.model_id

    model_id = 'dist_bert'
    batch_size = 12

    training_config = {
        'batch_size': batch_size
    }

    if is_cuda:
        try:
            torch.cuda.empty_cache()
        except:
            print("To release CUDA memory with error.")
            
    model_id = qa_model_dict.get(model_id)

    print("Start to Download pre-trained model")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForQuestionAnswering.from_pretrained(model_id)

    # put model to GPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    try:
        model.to(device)
    except Exception as e:
        print("When to load model to GPU with error: {}".format(e))

    if not model:
        print("Couldn't get the model to train, please check the model type!")
        sys.exit(-1)
    peft_training = False

    if peft_training:
        model = get_peft_model_lora_based(model)
        
    dataset = get_train_dataset(json_path='sample.json')
        
    train_model(model, tokenizer, dataset=dataset, output_dir=model_id + '_no_peft', training_config=training_config)

    # make one sample 
    question = "How many programming languages does BLOOM support?"
    context = "BLOOM has 176 billion parameters and can generate text in 46 languages natural languages and 13 programming languages."

    answer_str = get_model_prediction(question, context=context, model=model, tokenizer=tokenizer)
    # Tested is right.
    print("{}\n Get result: {}".format(question, answer_str))


