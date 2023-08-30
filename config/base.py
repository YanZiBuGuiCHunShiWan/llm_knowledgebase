from __future__ import annotations
from abc import abstractmethod
import asyncio
from typing import Dict,List,Literal
from langchain.schema import Document
from config.constant import FaissVectorConfig
from text_splitter.custom_splitter import CustomTextSplitter
from knowledge_base.es.es_builder import ElasticsearchScholar
from knowledge_base.vectorbase.Faiss_knowledgebase import FaissVectorScholar
from typing import Optional,Literal
__all__=[
    "BaseKnowledgeManager",
    "BaseTextSplitter",
    "BaseScholar"
]

class BaseKnowledgeManager:
    '''
    知识管理器：
        1.支持加载文档并由文档管理者处理文档。
        3.可以加载知识。
        4.可以插入或者删除知识。
    '''
    
    def __init__(self,es_scholar:ElasticsearchScholar,
                 vec_scholar:FaissVectorScholar,
                 text_splitter:CustomTextSplitter):
        self.es_scholar=es_scholar
        self.vec_scholar=vec_scholar
        self.text_splitter=text_splitter

    @abstractmethod
    async def add_text(self,text:List[Document]):
        raise NotImplementedError
    
    @abstractmethod
    async def delete_text(self,doc_index:int,index_naem:str):
        raise NotImplementedError
    
    @abstractmethod
    def show_index_samples(self):
        '''
        显示ElasticSearch和向量索引的样本数，二者最好相匹配。
        如果ElasticSearch加载A领域的文档，向量索引加载B领域的文档，那么二者样本数就不相同。
        '''
        raise NotImplementedError
    
class BaseTextSplitter:
    '''
    文本切割器：用于加载文档并且按照自定义方式切分
    
    '''
    def __init__(self,data_path:str):
        self.docs=[]
    
    @abstractmethod
    def load_documents(self):
        raise NotImplementedError
    
class BaseScholar:
    '''
    '''
    def __init__(self,config):
        self.config=config
    
    @abstractmethod
    def add_text(self,doclists:List[Document],index_name:str):
        raise NotImplementedError
    
    @abstractmethod
    def delete_text(self,doc_index:int,index_name:str):
        raise NotImplementedError
    
    @abstractmethod
    def drop_index(self,inde_name:str):
        raise NotImplementedError