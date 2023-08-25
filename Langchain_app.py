from langchain.chains import RetrievalQA
from langchain.prompts.prompt import PromptTemplate


class LangChainApp():
    
    def __init__(self) -> None:
        pass
        RetrievalQA.from_llm()
    
    def knowledge_base_answer(self,query:str):
        '''
        检索后再将相关结果给大模型，由大模型整理后回答
        '''
        pass
    
    
    
    def llm_answer(self,query:str):
        '''
        直接使用大模型回答
        '''
        pass
    
    
    
    