# used for NLP preprocessing that could be used for batch and new query
"""spacy pretrained model downlaod: 
python -m spacy download en_core_web_sm
"""
import spacy
import re
import os
import PyPDF2 as pdf
from pathlib import Path

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma, Redis
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from langchain.docstore.document import Document


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
        text = self._remove_symbols(text)
        tokens = self.tokenize(text)
        filtered_tokens = self.remove_stopwords(tokens)
        lemmas = self.lemmatize(filtered_tokens)
        preprocessed_text = " ".join(lemmas)
        preprocessed_text = self._filter_text(text=preprocessed_text)
        return preprocessed_text


class PDFExtract:
    # todo: whether or not to use NLP preprocesing of files? if use it, but the response won't be continues
    def __init__(self, nlp_processor=None) -> None:
        self.nlp_processor = NLPProcessor() if nlp_processor is None else nlp_processor
        
    @staticmethod
    def _get_pdf_file_list(pdf_file_path):
        files = [x for x in os.listdir(pdf_file_path) if x.lower().endswith('.pdf')]
        return files
    
    @staticmethod
    def _filter_paragraph(text, n_min_words=30):
        if len(text.split()) < n_min_words:
            text = ''
        return text
    
    def extract_pdf_file(self, pdf_file_path, file_name, dump_txt_file_path=None, save_mid_txt=True, return_txt_file_name=True):
        print("start to process file: {}".format(file_name))
        pdf_path = os.path.join(pdf_file_path, file_name)
        text_list = self.extract_paragraphs_from_pdf(pdf_path)
        text_list = [x for x in [self._filter_paragraph(text) for text in text_list] if x != '']

        if file_name.endswith('.pdf'):
            txt_file_name = file_name.replace('.pdf', '.txt')
            
        if save_mid_txt:
            if not dump_txt_file_path:
                dump_txt_file_path = pdf_file_path
            _dump_text_to_local(text_list, file_name=txt_file_name, txt_file_path=dump_txt_file_path)
            
        return txt_file_name
    
    def extract_files(self, pdf_file_path):
        files = self._get_pdf_file_list(pdf_file_path)
        for file_name in files:
            self.extract_pdf_file(pdf_file_path=pdf_file_path, file_name=file_name)
        
    @staticmethod
    def extract_paragraphs_from_pdf(pdf_path):
        paragraphs = []
        with open(pdf_path, 'rb') as file:
            pdf_reader = pdf.PdfReader(file)
            for page in pdf_reader.pages:
                text = page.extract_text()
                paragraphs.extend(text.split('\n\n'))
        
        # Filter out empty paragraphs
        paragraphs = [paragraph.strip() for paragraph in paragraphs if paragraph.strip()]
        return paragraphs
        
        
def _dump_text_to_local(txt_list, file_name, txt_file_path):
    if not os.path.exists(txt_file_path):
        os.makedirs(txt_file_path,exist_ok=True)
   
    dest_txt_path = os.path.join(txt_file_path, file_name)
    with open(dest_txt_path, 'w', encoding='utf-8') as f:
        for text in txt_list:
            f.write(text + '\n')


class VectorStore:
    def __init__(self, docs=None, em_model_id=None, persist_folder_name="chroma") -> None:
        self.persist_dic = persist_folder_name
        em_model_id = em_model_id if em_model_id else "all-MiniLM-L6-v2"
        self.embedding = HuggingFaceEmbeddings(model_name=em_model_id)
        if docs:
            # if init with docs, else load it from disk.
            print("Start to load files from texts.")
            if isinstance(docs[0], Document):
                self.vs = Chroma.from_documents(documents=docs, 
                                                embedding=self.embedding, 
                                                persist_directory=self.persist_dic)
            else:
                self.vs = Chroma.from_texts(texts=docs, 
                                            embedding=self.embedding, 
                                            persist_directory=self.persist_dic)
            self.vs.persist() # trigger dump.
        else:
            self.vs = Chroma(self.persist_dic,embedding_function=self.embedding)
    
    def similarity_search(self, query, k=3):
        # self._check_vs_exist()
        return self.vs.similarity_search(query=query, k=k)
    
    def as_retriever(self):
        return self.vs.as_retriever() 


class VectorStoreBase:
    def __init__(self, em_model_id=None) -> None:
        em_model_id = em_model_id if em_model_id else "all-MiniLM-L6-v2"
        self.embedding = HuggingFaceEmbeddings(model_name=em_model_id)
        self.index_name = "vs_index"
        
    def init_vs(self, docs):
        raise NotImplementedError("Please use subclass to call.")
        
    def _check_vs_exist(self):
        if not hasattr(self, 'vs'):
            raise RuntimeError('please init vs first')
    
    def add_one_doc_str(self, doc):
        self._check_vs_exist()
        if not isinstance(doc, str):
            raise ValueError("doc should be string")
        self.vs.add_texts([doc])
    
    def add_docs(self, docs):
        for doc in docs:
            self.add_one_doc_str(doc=doc)
    
    def similarity_search(self, query, k=3):
        self._check_vs_exist()
        return self.vs.similarity_search(query=query, k=k)
    
    def delete_index(self):
        try:
            self.vs.drop_index(index_name=self.index_name, delete_documents=True, redis_url=self.rds_url)
        except:
            pass
        
    def as_retriever(self):
        return self.vs.as_retriever()
    
    def from_documents(self, docs):
        self.vs.from_documents(docs, embedding=self.embedding)
        
        
# class ChromaStore(VectorStoreBase):
#     def __init__(self, em_model_id=None) -> None:
#         super().__init__(em_model_id)
        
#     def init_vs(self, docs):
#         self.vs = Chroma.from_texts(docs, 
#                                     embedding=self.embedding, 
#                                     collection_name=self.index_name)
#         return self.vs
        


class VectorStoreRedis(VectorStoreBase):
    def __init__(self, em_model_id=None) -> None:
        self.rds_url = "redis://localhost:6379"
        self.vs = Redis.from_existing_index(embedding=self.embedding, index_name=self.index_name, redis_url=self.rds_url)
    
    def init_vs(self, docs):
        self.vs = Redis.from_texts(docs, 
                                   embedding=self.embedding, 
                                   redis_url=self.rds_url, 
                                   index_name=self.index_name)
        return self.vs
   
    
 
class DocumentsProcess:
    def __init__(self, chunk_size=500, chunk_overlap=30) -> None:
        self.txt_spliter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    def load_document(self, txt_path):
        docs = TextLoader(txt_path, encoding='utf-8').load()
        texts = self.txt_spliter.split_documents(docs)
        texts = [x.page_content for x in texts]
        return texts
    
    
class ProcessPipeline:
    def __init__(self,) -> None:
        self.doc_process = DocumentsProcess()
        self.pdf_extract = PDFExtract()
        # self.vector_store = VectorStore()
    
    def _load_text_from_file(self, pdf_file_path, pdf_file_name):
        txt_file_name = self.pdf_extract.extract_pdf_file(pdf_file_path=pdf_file_path, file_name=pdf_file_name)
        txt_path = os.path.join(pdf_file_path, txt_file_name)
        docs = DocumentsProcess().load_document(txt_path=txt_path)
        return docs
        
    def process(self, pdf_file_path):
        pdf_files = [x for x in os.listdir(pdf_file_path) if x.endswith('.pdf')]
        texts_list = []
        for file_name in pdf_files:
            texts = self._load_text_from_file(pdf_file_path=pdf_file_path, pdf_file_name=file_name)
            texts_list.extend(texts)
        self.vector_store = VectorStore(texts_list)
        return self
        
    def query(self, query, k=3):
        return self.vector_store.similarity_search(query=query, k=k)
        

if __name__ == '__main__':
    pdf_file_path = '.\sample_contract'
   
    pipe = ProcessPipeline().process(pdf_file_path=pdf_file_path)

    query = "what is Agency Agreement?"
    res = pipe.query(query=query)
    print(res)
    
    for r in res:
        print(r.page_content)
        print(r.metadata)
        print('\n')