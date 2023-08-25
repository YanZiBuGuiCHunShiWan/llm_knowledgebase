import glob
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.schema import Document
from typing import Dict,List
from modelscope.pipelines import pipeline

class CustomTextSplitter:    
    def __init__(self):
        self.docs=[]
        
    def load_documents(self,doc_path,suffix:str="*.txt")->List[Document]:
        files=glob.glob(doc_path+"/"+suffix)
        for file in files:
            self.docs.append(TextLoader(file).load()[0])
        return self.docs
    
    def recursive_charactor_split(self,chunk_size: int,chunk_overlap: int,length_func:len,add_start_index:True):
        '''
        By default the characters it tries to split on are ["\n\n", "\n", " ", ""]
        length_function: how the length of chunks is calculated. Defaults to just counting number of characters, but it's pretty common to pass a token counter here
        chunk_size: the maximum size of your chunks 
        chunk_overlap: the maximum overlap between chunks. It can be nice to have some overlap to maintain some continuity between chunks
        '''
        textsplitter=RecursiveCharacterTextSplitter(
                # Set a really small chunk size, just to show.
                chunk_size = chunk_size,
                chunk_overlap  = chunk_overlap,
                length_function = length_func,
                add_start_index = add_start_index,
            )
        splitted_texts=textsplitter.split_documents(self.docs)
        return splitted_texts
    
    def semantic_split(self,text:str):
        p = pipeline(
            task="document-segmentation",
            model='damo/nlp_bert_document-segmentation_chinese-base',
            device="cuda:1")
        result = p(documents=text)
        sent_list = [i for i in result["text"].split("\n\t") if i]
        
        return sent_list
    
if __name__=="__main__":
    doc_path="data/financial_research_reports"
    splitter=CustomTextSplitter()
    splitter.load_documents(doc_path=doc_path)
    splitted_result=splitter.recursive_charactor_split(100,20,len,True)
    print(len(splitted_result))