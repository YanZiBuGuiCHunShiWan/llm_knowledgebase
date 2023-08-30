import streamlit as st
import numpy as np
import asyncio
from text_splitter.custom_splitter import CustomTextSplitter
from knowledge_base.load_knowledge import CustomKnowledgeManager
from knowledge_base.es.es_builder import ElasticsearchScholar
from knowledge_base.vectorbase.Faiss_knowledgebase import FaissVectorScholar
from config.constant import (ElasticsearchConfig,
                             FaissVectorConfig,
                             LancdbVectorConfig,
                             PROMPT_TEMPLATE)
from tradition_application import RetrievalLLm
from loguru import logger

st.set_page_config(page_title="大模型智能问答系统")


@st.cache_resource
def init_application():
    data_path="data/financial_research_reports"
    es_config=ElasticsearchConfig()
    vec_config=FaissVectorConfig()
    text_splitter=CustomTextSplitter()
    ES_scholar=ElasticsearchScholar(es_config)
    Vec_scholar=FaissVectorScholar(vec_config)
    knowledge_manager=CustomKnowledgeManager(ES_scholar,Vec_scholar,text_splitter)
    if type(knowledge_manager.vec_scholar)==FaissVectorScholar:
        knowledge_manager.Load_Faiss_knowledge(data_path=data_path)
    application=RetrievalLLm(knolwedge_manager=knowledge_manager,
                             llm_path="../ptm/chatglm2")
    application.load_mdoel()
    return application,knowledge_manager

def clear_chat_history():
    del st.session_state.messages


def parse_session_messages(use_openai:bool):
    if len(st.session_state.messages)>0:
        if not use_openai:
            chat_history=[]
            for index in range(0,len(st.session_state.messages),2):
                chat_history.append((st.session_state.messages[index]["content"],st.session_state.messages[index+1]["content"]))
            return chat_history
        else:
            return st.session_state.messages   
    else:
        return []

def init_chat_history():
    with st.chat_message("assistant", avatar='🤖'):
        st.markdown("您好，我是人工智能助手小陆，很高兴为您服务🥰")

    if "messages" in st.session_state:
        for message in st.session_state.messages:
            avatar = '🧑‍💻' if message["role"] == "user" else '🤖'
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
    else:
        st.session_state.messages = []

    return st.session_state.messages

def switch_knowledge_base():
    pass
            
if __name__=="__main__":
    st.title("小陆智能问答系统")
    message=init_chat_history()
    application,knowledgemanager=init_application()
    with st.sidebar:
        retrieval_on=st.checkbox("是否启用检索功能,✔为启用")
        print(retrieval_on)
        if retrieval_on:
            retrieval_approach = st.radio(
                                        "请选择您的检索方式",
                                        ["关键词检索", "向量检索", "联合检索"],
                                        captions = ["Elastic Search的BM25算法", "语义向量检索", "Elastic Search和语义检索两种方式的结合"])
            options = st.selectbox(
                    '请选择您要检索的知识库',
                    ('金融领域公司年报', '维基百科'))
        
        llm_model=st.radio(
            "请选择您的智能对话助手👉",
            ["Openai","Chatglm2"]
        )
        use_openai=True if llm_model=="Openai" else False
        st.markdown("### :blue[🌟更多配置🌟]")
        if retrieval_on:
            topk=st.slider("选择检索文本个数",3,15,10,step=1)
        max_history=st.slider("多轮对话记忆数",0,10,5,step=1)
        top_p=None
        top_k=None
        if not use_openai:
            top_p=st.slider("Top p",0.0,1.0,0.85,step=0.05)
            top_k=st.slider("Top k",1,80,50,step=1)
        temperature=st.slider("温度系数",0.1,1.0,0.7,step=0.05)
    if prompt :=st.chat_input("请在此输入您的问题"):
        current_history=parse_session_messages(use_openai)
        print(current_history)
        with st.chat_message("Human",avatar="🧑‍💻"):
            st.markdown(f":red[{prompt}]")
        message.append({"role": "user", "content": prompt})
        with st.chat_message("assistant",avatar="🤖"):
            ##############先检索#####################
            if retrieval_on:
                search_result_list=asyncio.run(application.search_text(query=prompt,
                                                            approach="vector",
                                                            topk=topk,
                                                            index_name=ElasticsearchConfig().index_name))
                logger.info(search_result_list)
                search_result="".join(search_result_list)
                logger.info("检索文本长度为：{}".format(len(search_result)))
                if not use_openai:
                    with st.empty():
                        for llm_answer,current_history[-max_history:] in  asyncio.run(application.knowledge_base_answer(query=prompt,
                                                                             use_openai=False,
                                                                             retrieval_result=search_result,
                                                                             prompt_template=PROMPT_TEMPLATE,
                                                                             history=current_history[-max_history:],
                                                                             temperature=temperature,
                                                                             top_k=top_k,
                                                                             top_p=top_p)):
                            query, llm_answer = current_history[-1]
                            st.write(llm_answer)
                else:
                    llm_answer=asyncio.run(application.knowledge_base_answer(query=prompt,
                                                                prompt_template=PROMPT_TEMPLATE,
                                                                retrieval_result=search_result,
                                                                use_openai=use_openai,
                                                                history=current_history[-max_history:],
                                                                temperature=temperature))
                    place_holder=st.empty()
                    place_holder.markdown(f":blue{llm_answer}")
            else:
                if not use_openai:
                    with st.empty():
                        for llm_answer,current_history[-max_history:] in application.llm_answer(query=prompt,
                                                                             use_openai=False,
                                                                             history=current_history[-max_history:],
                                                                             temperature=temperature,
                                                                             top_k=top_k,
                                                                             top_p=top_p):
                            query, llm_answer = current_history[-1]
                            st.write(llm_answer)
                else:
                    llm_answer=application.llm_answer(query=prompt,
                                                  use_openai=use_openai,
                                                  history=current_history[-max_history:],
                                                  temperature=temperature)
                    place_holder=st.empty()
                    place_holder.markdown(f":blue{llm_answer}")
        message.append({"role": "assistant", "content":llm_answer})
        st.button("清空对话", on_click=clear_chat_history)
    










