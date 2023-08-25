from __future__ import annotations

__all__=[
    "VectorConfig",
    "ElasticsearchConfig",
    "url",
    "headers",
    "PROMPT_TEMPLATE"
]


url = 'https://api.openai-sb.com/v1/chat/completions'
headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer sb-82b14c3f5c0aa16b295c9d81ad1e465af920edf0245a4d51'
    }

DOCUMENT_PATH="data/financial_research_reports"
CHUNK_SIZE=100
CHUNK_OVERLAP=20

PROMPT_TEMPLATE = """【指令】根据已知信息，简洁和专业的来回答问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题”，不允许在答案中添加编造成分，答案请使用中文。 

【已知信息】{context} 

【问题】{question}"""

class VectorConfig:
    '''
    '''
    embedding_model_name_or_path="/data/text2vec-base-chinese"
    vector_store_path="dump/financialfaiss"
    vector_load_path="dump/financialfaiss"
    document_path="data/finalcial_research_reports"
    device="cuda:1"

class ElasticsearchConfig:
    '''
    index_name:以'knowledge'作为前缀
    '''    
    ip_address="127.0.0.1:9200"
    index_name="knowledge-financial"
    username="username",
    password="password"
    
    