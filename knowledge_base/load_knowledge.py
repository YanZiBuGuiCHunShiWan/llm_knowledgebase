import os
from langchain.schema import Document
from knowledge_base.es.es_builder import Elasticsearch_Builder
from knowledge_base.vectorbase.create_embeddings import EmbeddingsKnowledgeLoader
from config.constant import ElasticsearchConfig,VectorConfig
from text_splitter.custom_splitter import CustomTextSplitter
from loguru import logger

def load_text():
    docs_path="data/financial_research_reports/"
    docs=[]
    for doc in os.listdir(docs_path):
        if doc.endswith('.txt'):
            f=open(f'{docs_path}/{doc}','r',encoding='utf-8')
        docs.append(Document(page_content=''.join(f.read().split()), metadata={"source": f'doc_id_{doc}'}))
    return docs

def load_local_knowledge(data_path):
    
    es_config=ElasticsearchConfig()
    vs_config=VectorConfig
    elastic_server=Elasticsearch_Builder(es_config)
    vs_server=EmbeddingsKnowledgeLoader(vs_config)
    
    ###############加载知识#####################
    my_text_splitter=CustomTextSplitter()
    my_text_splitter.load_documents(data_path)
    splitted_chunks=my_text_splitter.recursive_charactor_split(chunk_size=200,
                                                               chunk_overlap=20,
                                                               length_func=len,
                                                               add_start_index=True)
    
    
    elastic_server.drop_index(drop_index_name=es_config.index_name)
    elastic_server.add_text(splitted_chunks,index_name=es_config.index_name)
    vs_server.embed_and_store(splitted_chunks)
    logger.info("执行成功")



if __name__=="__main__":
    data_path="data/financial_research_reports"
    load_local_knowledge(data_path=data_path)