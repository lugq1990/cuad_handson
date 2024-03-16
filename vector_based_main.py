"""spacy pretrained model downlaod: 
python -m spacy download en_core_web_sm
"""
import spacy
import re
import os
import PyPDF2 as pdf
from pathlib import Path
import json
import sys
import time
import torch
import pandas as pd
import collections
from langchain_community.vectorstores.chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings

from transformers import AutoModel, AutoTokenizer
from langchain.llms import GPT4All
from langchain.chains import RetrievalQA
import sys
import time
from transformers import AutoModel, AutoTokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer
from langchain.embeddings import HuggingFaceEmbeddings

import textwrap
from transformers import LlamaForCausalLM, LlamaTokenizer


SYSTEM_PROMPT = """\
    You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""
SYSTEM_PROMPT = textwrap.dedent(SYSTEM_PROMPT).strip()

auth_token = "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"
model_name = "meta-llama/Llama-2-7b-chat-hf"

# change model_name to new one that to avoid permission error:
model_name = "daryl149/llama-2-7b-chat-hf"


os.environ['hugging_access_token'] = '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'


model_dic = {
    'embedding_id': "sentence-transformers/all-MiniLM-L6-v2",
    "gpt4all":{
        "model_id": "ggml-gpt4all-j-v1.3-groovy.bin"
    },
    "transformers":{
        'chatglm': "THUDM/chatglm-6b-int4",
        'gpt2': "gpt2",
        'llama2': 'meta-llama/Llama-2-7b-chat-hf'
    }
}


prompt_template = """"Scenario: You are a legal professional working for a law firm that deals with various types of contracts. Your firm has recently developed an advanced content search engine that utilizes AI language model, LLM, to assist in analyzing and extracting relevant information from legal contracts. If the content haven't been provided, then you should use your know logic to try to answer the query.

Content: {}

Query: what is the potential risk for {}
"""


class NLPProcessor:
    def __init__(self, model_name="en_core_web_sm"):
        self.nlp = spacy.load(model_name, disable=["parser", "ner"])
        
    def _remove_symbols(self, text):
         text = re.sub(r'[^\w]', ' ', text)
         text = re.sub(' +', ' ', text)
         return text

    def tokenize(self, text):
        doc = self.nlp(text)
        tokens = [token.text for token in doc]
        return tokens

    def remove_stopwords(self, tokens):
        doc = self.nlp(" ".join(tokens))
        filtered_tokens = [token.text for token in doc if not token.is_stop]
        return filtered_tokens

    def lemmatize(self, tokens):
        doc = self.nlp(" ".join(tokens))
        lemmas = [token.lemma_ for token in doc]
        return lemmas
    
    @staticmethod       
    def _filter_text(text, n_words=10):
        if len(text.split()) < n_words:
            return ''
        return text
    
    def preprocess_text(self, text):
        # text = self._remove_symbols(text)
        tokens = self.tokenize(text)
        filtered_tokens = self.remove_stopwords(tokens)
        lemmas = self.lemmatize(filtered_tokens)
        preprocessed_text = " ".join(lemmas)
        # preprocessed_text = self._filter_text(text=preprocessed_text)
        return preprocessed_text


def clause_extract(text, min_words=30):
    processor = NLPProcessor()
    text_list = [x.strip() for x in text.split('\n\n')]
    para_list = []
    for text in text_list:
        processed_text = processor.preprocess_text(text)
        if len(processed_text) > min_words:
            para_list.append(processed_text)
            
    return para_list


class VectorDB:
    def __init__(self, model_id=None) -> None:
        model_id = model_id if model_id else  "all-MiniLM-L6-v2"
        self.embedding = HuggingFaceEmbeddings(model_name=model_id)

    def _init_db(self, text_list):
        self.db = Chroma.from_texts(text_list, embedding=self.embedding)
        
    def query(self, query, topk=1):
        docs = self.db.similarity_search(query, k=topk)
        if len(docs):
            return docs[0].page_content


def _load_contract(file_name, file_path=None):
    if not file_path:
        file_path = 'contracts'
    with open(os.path.join(file_path, file_name), 'r', encoding='utf-8') as f:
        return f.read()


def _load_json(file_name):
    with open(file_name, 'r') as f:
        json_data = json.loads(f.read())
    return json_data



def format_prompt(history, message, system_prompt):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    prompt = f"{B_INST} {B_SYS}{system_prompt}{E_SYS} "
    for user_msg, asst_msg in history:
        user_msg = str(user_msg).strip()
        asst_msg = str(asst_msg).strip()
        prompt += f"{user_msg} {E_INST} {asst_msg} </s><s> {B_INST} "

    message = str(message).strip()
    prompt += f"{message} {E_INST} "
    return prompt


class LLama2:
    def __init__(self, model_name, auth_token, temperature=0.6, top_p=0.9, max_gen_len=4096) -> None:
        self.temperature = temperature
        self.top_p = top_p
        self.max_gen_len = max_gen_len
        self.model = LlamaForCausalLM.from_pretrained(
            model_name, use_auth_token=auth_token, load_in_4bit=True, device_map="auto"
        ).eval()
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name, use_auth_token=auth_token)
     
    def generate(self, prompt, history):
        prompt = format_prompt(history, prompt, SYSTEM_PROMPT)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        model_resp = self.model.generate(
            **inputs,
            max_new_tokens=self.max_gen_len,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        # get text from tokenizer
        model_resp = self.tokenizer.decode(model_resp[0], batched=True)
        
        # try to remove the prompt from the output
        model_resp = model_resp.replace(prompt, '')
        
        return model_resp



def init_db(contract_file):
    text = _load_contract(contract_file)
    para_list = clause_extract(text=text)
    

    db = VectorDB()
    db._init_db(text_list=para_list)
    return db


def process_one_clause(model, db, clause_name):
    """Should also provide the extracted clause context for the later use case that 
    we could get some insights that maybe not the model get error, but just the content 
    provided is not correct!
    """
    clause_info = {}
    # at least we should ensure that this key words should be in the text.
    query = clause_name  
    if query not in clause_sim_dict:
        print("The clause: {} is not supported! Stop".format(query))
        sys.exit(-1)
    else:
        similar_clauses = clause_sim_dict.get(query, [])
        similar_clauses.append(query)
        # to get the query, should also get some related clause names based on the dictionary.
        enhanced_query = ' & '.join(similar_clauses)
        
        related_para = db.query(query=enhanced_query)
        
        if len(related_para) > 0:
            # this prompt could be sent to the LLM
            prompt = prompt_template.format(related_para, query)
            resp = model.generate(prompt=prompt, history=[])
            
            # remove no used <s> from the response
            resp = resp.replace('<s>', '').strip()
            print(resp)
        else:
            resp = "Couldn't get related paragraph from the contract for clause: {}".format(query)
            print(resp)
        # add info to res.
        clause_info['clause_name'] =  clause_name
        clause_info['extracted_clause'] =  related_para
        clause_info['model_response'] =  resp
        return clause_info


def process_one_contract(model, contract_file, clause_list):
    """Process one contract at one time."""
    db = init_db(contract_file=contract_file)
    clause_list_result = []
    
    for clause in clause_list:
        clause_res = process_one_clause(model, db=db, clause_name=clause)
        clause_list_result.append(clause_res)
        
    return clause_list_result
    
          

def _dump_model_output(model_output_dict, raw_json_data_file_name='model_output_res.json', output_excel_file_name='model_output_with_rating.xlsx'):
    """Dump model output to disk for later use.
    
    For user validation excel, each sheet is the contract name, provide 2 cols: 
        extract_clause_correct: is extracted clause correct?
        is_model_analysis_useful: is model output is useful for user?

    Args:
        model_output_dict (json): a dict key with key name is contract name, value is a list of output res
        raw_json_data_file_name (str, optional): _description_. Defaults to 'model_output_res.json'.
        output_excel_file_name (str, optional): _description_. Defaults to 'model_output_with_rating.xlsx'.
    """
    if raw_json_data_file_name:
        print("Start to dump the model final output to disk: ")
        with open(raw_json_data_file_name, 'w') as f:
            f.write(json.dumps(model_output_dict))
    if output_excel_file_name:
        # dump json to excel for user
        with pd.ExcelWriter('output.xlsx') as writer:
            for k, v in model_output_dict.items():
                df = pd.DataFrame(v)
                df['extract_clause_correct'] = ''
                df['is_model_analysis_useful'] = ''
                df.to_excel(writer, k, index=False)



if __name__ == '__main__':
    "read the contrat and init the vectordb, get user query, and get similar paragraph, use this to do prompt constraction."  
    start_time = time.time()  
    try:
        # first time to release gpu 
        torch.cuda.empty_cache()
    except:
        print("To release CUDA memory with error.")

    contract_path = 'contracts'
    contract_list = os.listdir(contract_path)
    
    # get the predefined clause list.
    clause_sim_dict = _load_json('./similar_clause_dict.json')
    clause_list = list(clause_sim_dict.keys())

    # next step is to build LLM instance that could handle requst from user.
    model = LLama2(model_name=model_name, auth_token=auth_token)
          
    # Next step is to use this model to get model prediction for full list of contracts, for each clause will get a output
    # then just dump each of them into disk for user next step evaluation.
    model_output_dict = collections.defaultdict(list)

    for contract_file in contract_list:
        print("Start to process file: {}".format(contract_file))
        print('-'* 100)
        model_output_dict[contract_file] = []
        clause_list_result = process_one_contract(contract_file=contract_file, clause_list=clause_list)

    _dump_model_output(model_output_dict=model_output_dict)
    
    end_time = time.time()
    print("Full process takes: {} mins to process: {} contracts".format((end_time - start_time)/ 60), len(contract_list))
    