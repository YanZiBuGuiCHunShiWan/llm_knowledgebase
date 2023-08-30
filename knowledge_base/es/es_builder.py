from typing import List, Dict
import time,os
from langchain.schema import Document
from loguru import logger
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import ssl
from config.constant import ElasticsearchConfig
from tqdm import tqdm

class ElasticsearchScholar():
    
    def __init__(self,elastic_config:ElasticsearchConfig):
        '''
        requires elasticsearch version: 7.11.0
        '''
        self.es_config=elastic_config
        context=ssl._create_unverified_context()
        self.es=Elasticsearch(
            [elastic_config.ip_address],
            # http_auth=(elastic_config.username,
            #            elastic_config.password),
            scheme="http",
            ssl_context=context
        )
        
        if self.es.ping():
            logger.info("Instantiated Elastic search successfully.")
        else:
            raise Exception("Connecting error.")
        
    def __check_index(self,index_name:str):
        if not self.es.indices.exists(index=index_name):
                logger.info("该索引不存在")
    
    def add_text(self,doc_lists:List[Document],index_name:str):
        logger.info("elastic search start adding text.........")
        custom_mapping={
            "mappings":{
                "news_type":{
                    "properties":{
                        "text_spans":{
                            "type":"text",
                            "index":True,
                            "analyzer":"ik_max_word"
                        }
                    }
                }
            }
        }
        
        if not self.es.indices.exists(index=index_name):
            self.es.indices.create(index=index_name,body=custom_mapping,ignore=400)
            current_indices_nums= 0 #
        else:
            current_indices_nums=self.es.count(index=index_name)["count"]
        Actions=[]
        for index,content in tqdm(enumerate(doc_lists),total=len(doc_lists)):
            action={
                "_index":index_name,
                "_type":"text_contents",
                "_id":current_indices_nums+index,
                "_source":{
                    "text_spans":content.page_content
                }
            }
            Actions.append(action)
        
        success,_=bulk(self.es,Actions,index=index_name,raise_on_error=True)
        logger.info("Index :{} perform {} actions".format(index_name,success))
        
    def text_search(self,query_content:str,topk:int,index_name:str)->List[Document]:
        self.__check_index(index_name)
        topk=min(self.es.count(index=index_name)["count"],topk)
        dsl = {
            'query': {
                'match': {
                    'text_spans': query_content
                },
            },
            'size': topk,
        }
        result=self.es.search(index=index_name, body=dsl)
        result_lists=[]
        for matched_content in result["hits"]["hits"]:
            result_lists.append(Document(page_content=matched_content["_source"]["text_spans"]))
        return result_lists
    
    def show_current_index_info(self):
        indices = self.es.indices.get_alias().keys()
        valid_indices=[]
        for index in indices:
            if index.startswith("knowledge-"):
                valid_indices.append({index:self.es.count(index=index)["count"]})
        print(valid_indices)
        return  valid_indices
    
    def drop_index(self,drop_index_name):
        self.es.indices.delete(index=drop_index_name,ignore=[400,404])
        logger.info("删除索引操作执行完毕。")
        
    def delete_text(self,doc_index:int,index_name:str):
        self.es.delete(index=index_name,id=doc_index)
        logger.info("elastic search:索引 {} 删除文本操作执行成功".format(index_name))
    
if __name__=="__main__":
    docs_path="data/financial_research_reports/"
    docs=[]
    for doc in os.listdir(docs_path):
        if doc.endswith('.txt'):
        # print(doc)
        # loader = UnstructuredFileLoader(f'{docs_path}/{doc}', mode="elements")
        # doc = loader.load()
            f=open(f'{docs_path}/{doc}','r',encoding='utf-8')

        # docs.extend(doc)
        docs.append(Document(page_content=''.join(f.read().split()), metadata={"source": f'doc_id_{doc}'}))
    content_text=[doc.page_content for doc in docs]
    es_config=ElasticsearchConfig()
    es_server=ElasticsearchScholar(es_config)
    #es_server.add_text(content_text,es_config.index_name)
    #es_server.show_current_index()
    es_server.drop_index("knowledge-financial")
    #result=es_server.text_search(query_content="燃料电池",topk=10,index_name="knowledge-financial")
    #print(result)