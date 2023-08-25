import numpy as np
from knowledge_base.vectorbase.create_embeddings import EmbeddingsKnowledgeLoader
from elasticsearch import Elasticsearch
from knowledge_base.es.es_builder import Elasticsearch_Builder
from langchain.vectorstores import FAISS
from config.constant import ElasticsearchConfig,VectorConfig
from langchain.schema import Document

from typing import List

class CustomRetriever:
    
    def __init__(self,es_obj:Elasticsearch_Builder,vec_obj:EmbeddingsKnowledgeLoader):
        self.es=es_obj
        self.vecstore=vec_obj.vecstore
        self.transformer=vec_obj.transformer

    def vector_retrieve(self,query:str)->List[Document]:
        text_embedding=self.transformer.embed_query(query)
        result=self.vecstore.similarity_search_by_vector(text_embedding)
        return result
        
    def BM25_retrieve(self,query:str,index_name:str)->List[Document]:
        result=self.es.text_search(query_content=query,topk=10,index_name=index_name)
        return result
    
    def merge_retrieve(self,query:str)->List[Document]:
        pass

def load_searcher():
    es_config=ElasticsearchConfig()
    vs_config=VectorConfig()
    es_searcher=Elasticsearch_Builder(es_config)
    vs_searcher=EmbeddingsKnowledgeLoader(vs_config)
    return es_searcher,vs_searcher

if __name__=="__main__":
    
    es_config=ElasticsearchConfig()
    vs_config=VectorConfig()
    es_searcher=Elasticsearch_Builder(es_config)
    vs_searcher=EmbeddingsKnowledgeLoader(vs_config)
    vs_searcher.load_embeddings(load_path=vs_config.vector_load_path)
    my_retriever=CustomRetriever(es_obj=es_searcher,vec_obj=vs_searcher)
    
    result=my_retriever.vector_retrieve("休闲游戏广告联盟")
    print(result[0].page_content[:30])
    text_result=my_retriever.BM25_retrieve("休闲游戏广告联盟",index_name=es_config.index_name)
    print(text_result[0].page_content[:30])
    
    