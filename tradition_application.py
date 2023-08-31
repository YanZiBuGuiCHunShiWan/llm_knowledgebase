import torch,os
from transformers import AutoTokenizer,AutoModel,AutoModelForCausalLM,BitsAndBytesConfig
from text_splitter.custom_splitter import CustomTextSplitter
from loguru import logger
import requests,aiohttp
import asyncio
from typing import List,Dict
from knowledge_base.load_knowledge import CustomKnowledgeManager
from config.constant import ElasticsearchConfig,FaissVectorConfig,URL,HEADERS,PROMPT_TEMPLATE
from knowledge_base.load_knowledge import CustomKnowledgeManager
from knowledge_base.es.es_builder import ElasticsearchScholar
from knowledge_base.vectorbase.Faiss_knowledgebase import FaissVectorScholar

class RetrievalLLm(object):
    def __init__(self,knolwedge_manager:CustomKnowledgeManager,llm_path:str):
        self.knowledge_manager=knolwedge_manager
        self.llm_path=llm_path
        
    @staticmethod
    async def async_post_request(url:str, headers:dict,data:dict):
        async with aiohttp.ClientSession() as session:
            async with session.post(url,headers=headers,json=data) as response:
                return await response.json()

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
    
    async def search_text(self,query:str,approach:str,topk:int,index_name:str)->List[str]:
        result = await self.knowledge_manager.search_text(query=query,approach=approach,topk=topk,index_name=index_name)
        return result
    
    
    async def local_model_answer(self,query,history):
        if self.tokenizer.__class__.__name__=="ChatGLMTokenizer":
            return await asyncio.get_event_loop().run_in_executor(None,self.llm_model.stream_chat,self.tokenizer,
                                            query,
                                            history
                                            )
    
    async def openai_answer(self,query,history):
            data = {
            'model': 'gpt-3.5-turbo',
            'messages': history+[{'role': 'user', 'content':query}],
            'temperature': 0.7
            }
            try:
                openai_result =  await self.async_post_request(url=URL,headers=HEADERS,data=data)
                response=openai_result["choices"][0]["message"]["content"]
            except Exception as err:
                print(err)
                print(openai_result)
                response="抱歉，您发起的请求因网络故障导致失败，请您再次输入您的内容~"
            return response
        
    async def retrieve_reply(self,query,history,approach,topk,index_name,use_openai,prompt_template):
            retrieval_result = await self.search_text(query,approach,topk,index_name)
            model_input=prompt_template.format(context=retrieval_result,question=query)
            if use_openai:
                return await self.openai_answer(model_input,history)
            else:
                return await self.local_model_answer(model_input,history)
            
        
    def clear_history(self):
        self.history=[]

if __name__=="__main__":
    ####################################  work flow  ########################################
    ####################################   step1  ###########################################
    '''split text into small chunks'''
    
    data_path="data/financial_research_reports"
    es_config=ElasticsearchConfig()
    vec_config=FaissVectorConfig()
    text_splitter=CustomTextSplitter()
    ES_scholar=ElasticsearchScholar(es_config)
    Vec_scholar=FaissVectorScholar(vec_config)
    knowledge_manager=CustomKnowledgeManager(ES_scholar,Vec_scholar,text_splitter)
    
    '''当向量检索工具是Faiss时我们需要把索引从磁盘读取到内存中'''
    knowledge_manager.Load_Faiss_knowledge(data_path="data/financialfaiss",index_name=es_config.index_name)

    
    ###################################    step2   ###########################################
    '''input a query,  the RetrievalLLm will first retrieve chunks related to query then feed them into \
        the large language model(llm) ,finally the llm will read the chunks and generate answer.'''
    
    application=RetrievalLLm(knolwedge_manager=knowledge_manager,
                             llm_path="../ptm/chatglm2")
    application.load_mdoel()
    #prompt_text="""已知信息:{}。用户问题如下:{}，请你咨询阅读提供的信息，记住关键数字，并进行一定程度地推理。如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"。"""
    search_result_list=asyncio.run(application.search_text(query="雅居乐集团2017-2018年销售面积约为多少万平方米？",
                                                           approach="vector",
                                                           topk=5,
                                                           index_name=es_config.index_name))
    search_result="".join(search_result_list)
    logger.info("检索结果:{}".format(search_result))
    llm_answer=asyncio.run(application.retrieve_reply(query="雅居乐集团2017-2018年销售面积约为多少万平方米？",
                                                            history=[],
                                                            approach="es",
                                                            prompt_template=PROMPT_TEMPLATE,
                                                            use_openai=True,
                                                            topk=3,
                                                            index_name=es_config.index_name))
    print(llm_answer)
    history=[]
    for answer,history in asyncio.run(application.retrieve_reply(query="雅居乐集团2017-2018年销售面积约为多少万平方米？",
                                                            history=[],
                                                            approach="vector",
                                                            prompt_template=PROMPT_TEMPLATE,
                                                            use_openai=False,
                                                            topk=3,
                                                            index_name=es_config.index_name)):
        query,answer=history[-1]
    print(answer)