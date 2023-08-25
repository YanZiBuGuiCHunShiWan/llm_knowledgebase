import torch,os
from transformers import AutoTokenizer,AutoModel,AutoModelForCausalLM,BitsAndBytesConfig
from tools.retriever import CustomRetriever
from text_splitter.custom_splitter import CustomTextSplitter
from loguru import logger
import requests
from typing import List,Dict
from tools.retriever import load_searcher
from langchain.schema import Document
from config.constant import url,headers
from config.constant import PROMPT_TEMPLATE

class RetrievalLLm(object):
    def __init__(self,retreiver:CustomRetriever,llm_path:str) -> None:
        self.retriever=retreiver
        self.llm_path=llm_path
        self.history=[]
        
    def load_mdoel(self):
        self.tokenizer=AutoTokenizer.from_pretrained(self.llm_path,trust_remote_code=True)
        if self.tokenizer.__class__.__name__=='ChatGLMTokenizer':
            AutoLoader=AutoModel
        else:
            AutoLoader=AutoModelForCausalLM
        self.llm_model=AutoLoader.from_pretrained(self.llm_path,
                                                  trust_remote_code=True,
        device_map=torch.cuda.current_device())   
        self.llm_model.eval()

    def text_retrieve(self,query:str,index_name:str)->List[Document]:
        '''
        methods:choices= ["faiss","bm25","merge"]
        self.retriever 执行 retrieve
        '''
        result=self.retriever.BM25_retrieve(query=query,index_name=index_name)
        return result
    
    def semantic_retrieve(self,query:str)->List[Document]:
        result=self.retriever.vector_retrieve(query=query)
        return result
    
    def filter_result(self):
        
        '''
        filter some unuseful results by some specific rules.
        '''
        pass
    
    def rank_result(self):
        '''
        rank topk documents with a cross encoder.
        '''
        pass
    
    def knowledge_base_answer(self,query:str,prompt_template:str,index_name:str,methods="bm25",use_openai=False):
        '''
        1.把query放进prompt
        2.再和检索得到的结果一起放进大模型
        3.得到最终结果
        '''
        assert methods in ["bm25","semantic"]
        if methods=="bm25":
            retrieval_result=self.text_retrieve(query=query,index_name=index_name)
        elif methods=="semantic":
            retrieval_result=self.semantic_retrieve(query=query)
        retrieval_result="".join([doc.page_content for doc in retrieval_result][:2])
        print(retrieval_result)
        prompt_input=prompt_template.format(context=retrieval_result,question=query)
        if not use_openai:
            response, self.history = self.llm_model.chat(self.tokenizer, prompt_input, history=[])
        else:
            data = {
            'model': 'gpt-3.5-turbo',
            'messages': [{'role': 'user', 'content':prompt_input}],
            'temperature': 0.7
            }
            openai_result=requests.post(url, headers=headers, json=data)
            json_result=openai_result.json()
            response=json_result["choices"][0]["message"]["content"]
        return response
    
    def web_search(self,query):
        pass
    
    def llm_answer(self,query,prompt_template,web_search=False,use_openai=False):
        if not web_search:
            response, self.history = self.llm_model.chat(self.tokenizer, prompt_template.format(query), history=[])
        return response
    
    def clear_history(self):
        self.history=[]

if __name__=="__main__":
    ####################################  work flow  ########################################
    ####################################   step1  ###########################################
    '''split text into small chunks'''
    data_path="data/financial_research_reports"
    my_text_splitter=CustomTextSplitter()
    my_text_splitter.load_documents(data_path)
    splitted_chunks=my_text_splitter.recursive_charactor_split(chunk_size=200,
                                                               chunk_overlap=20,
                                                               length_func=len,
                                                               add_start_index=True)
    ####################################   step2  ###########################################
    '''store these chunks into elastic search or convert them to sentence embeddings.'''
    es_searcher,vec_searcher=load_searcher()
    #es_searcher.drop_index(drop_index_name=es_searcher.es_config.index_name)
    #es_searcher.add_text(splitted_chunks,index_name=es_searcher.es_config.index_name)
    es_searcher.show_current_index_info()
    if os.path.exists(vec_searcher.vector_config.vector_load_path):
        vec_searcher.load_embeddings(vec_searcher.vector_config.vector_load_path)
    else:
        vec_searcher.embed_and_store(splitted_chunks)
    
    ###################################    step3  ############################################
    '''instantiate  a custom retriever and the RetrievalLLm'''
    llm_path="../ptm/chatglm2"
    custom_retriever=CustomRetriever(es_searcher,vec_searcher)
    llm_retriever=RetrievalLLm(retreiver=custom_retriever,llm_path=llm_path)
    
    ###################################    step4   ###########################################
    '''input a query,  the RetrievalLLm will first retrieve relevant chunks to query then feed them into \
        the large language model(llm) ,finally the llm will read the chunks and generate answer.'''
    
    llm_retriever.load_mdoel()
    #prompt_text="""已知信息:{}。用户问题如下:{}，请你咨询阅读提供的信息，记住关键数字，并进行一定程度地推理。如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"。"""
    
    retrieval_result=llm_retriever.knowledge_base_answer(query="雅居乐集团2017-2018年销售面积约为多少万平方米？",
                                                         prompt_template=PROMPT_TEMPLATE,
                                                         index_name=es_searcher.es_config.index_name,
                                                         use_openai=True)                            
    # #search_result=llm_retriever.llm_answer(query="雅居乐集团 2013-2018 年平均竣工面积约为？",prompt_template="{}")
    # #print(search_result)
    print(retrieval_result)