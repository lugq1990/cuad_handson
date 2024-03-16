import os
import json
from transformers import AutoModel, AutoTokenizer
from langchain.llms import GPT4All
from langchain.chains import RetrievalQA
import sys
import time
from transformers import AutoModel, AutoTokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer
from langchain.embeddings import HuggingFaceEmbeddings
import torch
# add this to avoid import error.
sys.path.append(r'D:\work\code_related\my_codes\llm_implement')
# from chat_llm.preprossing import VectorStore

os.environ['hugging_access_token'] = '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'


def init_env():
    with open('openai.keys', 'r') as f:
        data = f.read()
    
    dic = json.loads(data)
    for k, v in dic.items():
        os.environ[k] = v

init_env()

prompt_template = """"Scenario: You are a legal professional working for a law firm that deals with various types of contracts. Your firm has recently developed an advanced content search engine that utilizes AI language model, LLM, to assist in analyzing and extracting relevant information from legal contracts. If the content haven't been provided, then you should use your know logic to try to answer the query.

Content: {}

Query: {}
"""


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




class LLAMA2:
    def __init__(self, model_name, auth_token=None, temperature=0.6, top_p=0.9, max_gen_len=4096) -> None:   
        self.temperature = temperature
        self.top_p = top_p
        self.max_gen_len = max_gen_len
        if not auth_token:
            auth_token = os.environ['hugging_access_token']
        self.model = LlamaForCausalLM.from_pretrained(
            model_name, use_auth_token=auth_token, load_in_4bit=True, device_map="auto"
        ).eval()
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name, use_auth_token=auth_token)
        
    def generate(self, query):
        return self.model.generate(
            **query,
            max_new_tokens=self.max_gen_len,
            temperature=self.temperature,
            top_p=self.top_p
        )
        


class LLMModel:
    def __init__(self, model_id, embedding_id=None, embedding_device=None) -> None:
        self.model_id = model_id
        self.embedding_id = embedding_id if embedding_id else model_dic.get('embedding_id')
        if not embedding_device:
            if torch.cuda.is_available():
                embedding_device = 'gpu'
        self.embedding_device = embedding_device if not embedding_id else 'cpu'
        
    def load_llm_model(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            self.llm = AutoModel.from_pretrained(self.model_id, trust_remote_code=True).half().cuda()        
            print("Model have been loaded!")  
        except:
            self.tokenizer = None
            self.llm = None
    
    def _load_embeddings(self):
        self.embedding = HuggingFaceEmbeddings(self.embedding_id, 
                                               model_kwargs={'device':self.embedding_device})


class ChatModel:
    def __init__(self, use_gpt4all=False, model_name='llama2') -> None:
        self.use_gpt4all = use_gpt4all
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cpu':
            print("Noted: Using CPU is slow.")
        if use_gpt4all:
            try:
                from gpt4all import GPT4All
            except:
                raise ImportError("couldn't import gpt4all, please install it first.")
            # TODO: Provide the path of gpt4all model.
            model_path = r"C:\Users\MLoong\.cache\gpt4all\ggml-gpt4all-j-v1.3-groovy.bin"
            self.llm = GPT4All(model=model_path, n_batch=8,verbose=True, backend='gptj')
            self.qa_chain = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff", retriever=self.vectore_store.as_retriever(), return_source_documents=True)
        else:
            self._get_model_tokenizer(model_name=model_name)
        # self.vectore_store = VectorStore()
        

    def _get_model_tokenizer(self, model_name):
        trans_dic = model_dic.get('transformers')
        model_id = trans_dic.get(model_name)
        if model_name == 'chatglm':
            self.llm = AutoModel.from_pretrained(model_id, trust_remote_code=True).half().to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        elif model_name == 'llama2':
            llama_model = LLAMA2(model_name=model_id)
            self.llm = llama_model.model
            self.tokenizer = llama_model.tokenizer
            
            
    def _construct_prompt(self, query, k=3):
        similary_content = self.vectore_store.similarity_search(query=query, k=k)
        if len(similary_content) > 0:
            content_list = [doc.page_content for doc in similary_content]
        else:
            print("Couldn't get simiary content from docs, use baes model functionality.")
            content_list = []
            
        prompt = prompt_template.format('\n'.join(content_list), query)
        return prompt
    
    def _llm_model(self, query):
        tokens = self.tokenizer(query, return_tensors='pt').to(self.device)
        with torch.no_grad():
          outputs = self.llm(**tokens)
        logits = outputs['last_hidden_state'][0]
        pred = torch.argmax(logits, dim=-1).numpy()
        # conver to words
        pred_words = self.tokenizer.decode(pred, batched=True)
        pred_words
        return pred_words
    
    def model_pred(self, query):
        start_time = time.time()
        prompt = query
        model_output = self._llm_model(prompt)
        print("AI system: ", model_output)
        
    def chat(self):
        while True:
            query = input("\nQuery:")
            prompt = query
            start_time = time.time()
            # load query similary content and put them into prompt
            # prompt = self._construct_prompt(query=query)
            # print("Get prompt:", prompt)
            if self.use_gpt4all:
                res = self.qa_chain(prompt)
                model_output = res['result']
                print("AI system: ", model_output)
                source_info = res['source_documents']
                try:
                    for doc in source_info:
                        print("Doc file path:", doc.metadata['source'])
                        print("Doc id: ", doc.metadata['id'])
                except:pass
            else:
                model_output = self._llm_model(prompt)
                print("AI system: ", model_output)
            end_time = time.time()
            print("Current chat takes {} seconds.".format(end_time - start_time))
        
        
if __name__ == "__main__":
    chat_model = ChatModel(use_gpt4all=False, model_name='chatglm')
    
    chat_model.chat()