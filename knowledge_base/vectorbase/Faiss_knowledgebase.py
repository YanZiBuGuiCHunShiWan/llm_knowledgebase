import os,torch,shutil
from typing import List,Dict,Literal
from langchain.schema import Document
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from config.constant import FaissVectorConfig
from loguru import logger

class FaissVectorScholar():
    def __init__(self,config:FaissVectorConfig):
        self.vector_config=config
        self.transformer=HuggingFaceEmbeddings(model_name=self.vector_config.embedding_model_name_or_path,
                                               model_kwargs={"device":torch.cuda.current_device()})
        self.vecstore=None

    def embed_and_store(self,doc_lists:List[Document]):
        logger.info("Faiss：开始编码文档........")
        self.vecstore=FAISS.from_documents(doc_lists,self.transformer)
        self.vecstore.save_local(self.vector_config.vector_store_path)
        logger.info("向量知识库创建成功,数据：{}条。".format(len(doc_lists)))
        
    def add_text(self,doclists:List[Document]):
        logger.info("Faiss：插入新的文档........")
        self.vecstore.add_documents(doclists)
        self.vecstore.save_local(self.vector_config.vector_store_path)
    
    def delete_text(self,doc_index:int):
        self.vecstore.delete([self.vecstore.index_to_docstore_id[doc_index]])
        logger.info("Faiss：指定文档删除成功。")
        self.vecstore.save_local(self.vector_config.vector_store_path)
        
    def drop_index(self,path:str):
        try:
            shutil.rmtree(path)
            logger.info("本地索引文件删除成功")
        except Exception as err:
            logger.info("文件夹不存在。")
            
    def text_search(self,query:str,topk:int):    
        result=self.vecstore.similarity_search(query,k=topk)
        return result
    
    
if __name__=="__main__":
    vectorconfig=FaissVectorConfig()
    docs_path="data/financial_research_reports/"
    docs=[]
    for doc in os.listdir(docs_path):
        if doc.endswith('.txt'):
            f=open(f'{docs_path}/{doc}','r',encoding='utf-8')

        # docs.extend(doc)
        docs.append(Document(page_content=''.join(f.read().split()), metadata={"source": f'doc_id_{doc}'}))
    knowledge=FaissVectorScholar(vectorconfig)
    knowledge.clear(path=vectorconfig.vector_store_path)
    #knowledge.embed_and_store(docs)
    #knowledge.load_embeddings(load_path=vectorconfig.vector_load_path)
    #print(type(knowledge.vecstore))
    #knowledge.add_text(docs)
    #print(knowledge.vecstore.index)
    #print(knowledge.vecstore.index.ntotal)