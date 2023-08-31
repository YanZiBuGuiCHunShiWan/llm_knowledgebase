import os
import asyncio
from typing import List
from langchain.schema import Document
from knowledge_base.es.es_builder import ElasticsearchScholar
from knowledge_base.vectorbase.Faiss_knowledgebase import FaissVectorScholar
from config.constant import ElasticsearchConfig,FaissVectorConfig
from config.func import async_timer_decorator,timer_decorator
from config.base import BaseKnowledgeManager
from text_splitter.custom_splitter import CustomTextSplitter
from loguru import logger
from langchain.vectorstores import FAISS

class CustomKnowledgeManager(BaseKnowledgeManager):
    
    @timer_decorator
    def Load_Faiss_knowledge(self,data_path,index_name):
        assert type(self.vec_scholar[index_name])==FaissVectorScholar,"向量库不为Faiss"
        try:
            self.vec_scholar[index_name].vecstore=FAISS.load_local(folder_path=self.vec_scholar[index_name].vector_config.vector_store_path,
                                                       index_name="index",
                                                       embeddings=self.vec_scholar[index_name].transformer)
            logger.info("Faiss：加载本地向量库成功。")
        except Exception as  err:
            logger.info("Faiss：加载本地文件失败，为您重新生成")
            loaded_docs=self.text_splitter.load_documents(data_path)
            splitted_chunks=self.text_splitter.recursive_charactor_split(loaded_docs,
                                                                         chunk_size=200,
                                                                        chunk_overlap=20,
                                                                        length_func=len,
                                                                        add_start_index=True)
            if len(splitted_chunks)>=100000:
                logger.warning("分块文档超过十万条，您将会等待很长时间。")
            self.vec_scholar[index_name].embed_and_store(splitted_chunks)

        
    @async_timer_decorator    
    async def delete_text(self, doc_id:int,approach:str,index_name:str):
        assert approach in ["es","vector","both"]
        if approach =="both":
            loop = asyncio.get_event_loop()
            tasks = [
                loop.run_in_executor(None, self.es_scholar.delete_text, doc_id,index_name),
                loop.run_in_executor(None, self.vec_scholar[index_name].delete_text,doc_id)
            ]
            await asyncio.wait_for(asyncio.gather(*tasks),timeout=60)
        elif approach=="es":
            await asyncio.get_event_loop().run_in_executor(None,
                                                           self.es_scholar.delete_text,
                                                           doc_id,
                                                           index_name)
        else:
            await asyncio.get_event_loop().run_in_executor(None,
                                                           self.vec_scholar[index_name].delete_text,
                                                           doc_id)

        
    
    @async_timer_decorator
    async def add_text(self, text_or_path: List[Document],approach:str,index_name:str):
        assert approach in ["es","vector","both"]
        '''
        '''
        if approach =="both":
            loop = asyncio.get_event_loop()
            tasks = [
                loop.run_in_executor(None, self.es_scholar.delete_text, text_or_path,index_name),
                loop.run_in_executor(None, self.vec_scholar[index_name].delete_text,text_or_path)
            ]
            await asyncio.wait_for(asyncio.gather(*tasks),timeout=60)
        elif approach=="es":
            await asyncio.get_event_loop().run_in_executor(None,
                                                           self.es_scholar.add_text,
                                                           text_or_path,
                                                           index_name)
        else:
            await asyncio.get_event_loop().run_in_executor(None,
                                                           self.vec_scholar[index_name].add_text,
                                                           text_or_path)
        

    @async_timer_decorator
    async def search_text(self,query:str,approach:str,index_name:str,topk:int)->List[str]:
        assert approach in ["es","vector","both"]
        if approach =="both":
            loop = asyncio.get_event_loop()
            tasks = [
            loop.run_in_executor(None,self.es_scholar.text_search,query,topk,index_name),
            loop.run_in_executor(None,self.vec_scholar[index_name].text_search,query,topk)
            ]
            
            es_result,vec_result = await asyncio.wait_for(asyncio.gather(*tasks),timeout=60)
            es_result=[page.page_content for page in es_result]
            vec_result=[page.page_content for page in vec_result]
            result=list(set(es_result+vec_result))
            
        elif approach=="es":
            result = await asyncio.get_event_loop().run_in_executor(None,
                                                                    self.es_scholar.text_search,
                                                                    query,topk,
                                                                    index_name)
            result=[page.page_content for page in result]
        else:
            result = await asyncio.get_event_loop().run_in_executor(None,
                                                                    self.vec_scholar[index_name].text_search,
                                                                    query,topk)
            result=[page.page_content for page in result]
        return result


if __name__=="__main__":
    data_path="data/financial_research_reports"
    es_config=ElasticsearchConfig()
    vs_config= FaissVectorConfig()
    elastic_server=ElasticsearchScholar(es_config)
    vector_server=FaissVectorScholar(vs_config)
    ###############加载知识#####################
    my_text_splitter=CustomTextSplitter()
    manager=CustomKnowledgeManager(es_scholar=elastic_server,
                                   vec_scholar=vector_server,
                                   text_splitter=my_text_splitter)
    