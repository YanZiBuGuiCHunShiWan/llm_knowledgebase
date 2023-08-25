import os,torch
from typing import List,Dict,Literal
from langchain.schema import Document
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from config.constant import VectorConfig
from loguru import logger



class EmbeddingsKnowledgeLoader():
    
    def __init__(self,vector_config:VectorConfig):
        self.vector_config=vector_config
        self.transformer=HuggingFaceEmbeddings(model_name=self.vector_config.embedding_model_name_or_path,
                                               model_kwargs={"device":torch.cuda.current_device()})
        self.vecstore=None

    def embed_and_store(self,doclists:List[Document]):
        self.vecstore=FAISS.from_documents(doclists,self.transformer)
        self.vecstore.save_local(self.vector_config.vector_store_path)
        logger.info("向量知识库创建成功。")
    
    def load_embeddings(self,load_path:str):
        if os.path.exists(load_path):
            try:
                self.vecstore=FAISS.load_local(folder_path=load_path,index_name="index",embeddings=self.transformer)
                logger.info("向量知识库加载成功。")
            except:
                raise Exception("路径下文件不齐全，请您重新生成。")
        else:
            raise Exception("尚未创建文件，请先存储embeddings。")
        
    def insert_knowledge(self):
        self.vecstore.add_embeddings()
        pass
    
    def delete_knowledge(self,doc_idnex:int):
        self.vecstore.delete([self.vecstore.index_to_docstore_id[doc_idnex]])
        logger.info("文档删除成功")
        
if __name__=="__main__":
    vectorconfig=VectorConfig()
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
    knowledge=EmbeddingsKnowledgeLoader(vectorconfig)
    knowledge.embed_and_store(docs)
    #knowledge.load_embeddings(load_path=vectorconfig.vector_load_path)
    print(type(knowledge.vecstore))
