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

from langchain_community.vectorstores.chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings


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
        model_id = model_id if not model_id else  "all-MiniLM-L6-v2"
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
    with open(os.path.join(file_path, file_name), 'r') as f:
        return f.read()


def _load_json(file_name):
    with open(file_name, 'r') as f:
        json_data = json.loads(f.read())
    return json_data


# if __name__ == '__main__':
    "read the contrat and init the vectordb, get user query, and get similar paragraph, use this to do prompt constraction."
contract_path = 'contracts'
contract_file = os.listdir(contract_path)[0]

text = _load_contract(contract_file)
para_list = clause_extract(text=text)
clause_sim_dict = _load_json('./similar_clause_dict.json')

db = VectorDB()
db._init_db(text_list=para_list)

# this should be clause name, 
# todo:at least we should ensure that this key words should be in the text.
query='Notice Period to Terminate Renewal'
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
    else:
        print("Couldn't get related paragraph from the contract for clause: {}".format(query))
        sys.exit(-1)
    # next step is to build LLM instance that could handle requst from user.


   