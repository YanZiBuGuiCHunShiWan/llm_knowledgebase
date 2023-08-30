from __future__ import annotations
import os,torch,shutil
import lancedb
from typing import List,Dict,Literal
from langchain.schema import Document
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from config.constant import LancdbVectorConfig
from loguru import logger


class LancdbVectorScholar():
    def __init__(self,config:LancdbVectorConfig):
        self.vector_config=config
        self.transformer=HuggingFaceEmbeddings(model_name=self.vector_config.embedding_model_name_or_path,
                                               model_kwargs={"device":torch.cuda.current_device()})
        self.vecstore=lancedb.connect("/data/lancedb").open_table(self.vector_config.table_name)
        self.table=self.vecstore.create_table(
                self.vector_config.table_name,
                data=[
                    {
                "vector": self.transformer.embed_query("Hello World"),
                "text": "Hello World",
                "id": "1",
                    }
                    ],
        mode="overwrite",
        )
        
        
        
    def create_table(self):
        pass
        
    def add_text(self, doclists: List[Document],table_name:str):
        return super().add_text(doclists)
    
    
    def delete_text(self, doc_index: int,table_name:str):
        return super().delete_text(doc_index)
    
        
    def drop_index(self, path: str):
        return super().drop_index(path)
    

