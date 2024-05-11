#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 16:01:51 2024

@author: galahassa
"""

# Load our emails deta using a json loader : JSONLoader

from langchain_community.document_loaders import JSONLoader
def metadata_func(record: dict, metadata: dict) -> dict:

    metadata["SenderName"] = record.get("SenderName")
    metadata["Subject"] = record.get("Subject")
    metadata["SenderEmail"] = record.get("SenderEmail")
    
    return metadata

loader = JSONLoader(
    file_path='./data/emails_2024_05_08_134553.json',
    jq_schema='.emails[]',
    text_content=False,
    metadata_func=metadata_func)


docs = loader.load()

# Split our documents into small chunks of text so that searching gets more efficient

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
splits = text_splitter.split_documents(docs)

# Use an embeddings model to transform our emails data into vectors
# Shortly explain what vectors are
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Create the vectors store using Chroma

from langchain_chroma import Chroma
vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_function)

# Gets the VectorStoreRetriever to search through our vectors database
retriever = vectorstore.as_retriever()

# BUILDING OUR CHAIN 

def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)

# 
retrieved_emails_data = retriever | format_docs

# Creating our prompts variable
from langchain_core.runnables import RunnablePassthrough
task_input = RunnablePassthrough()
prompt_inputs = {"context": retrieved_emails_data, "task": task_input}


# Defining our prompt
from langchain_core.prompts import PromptTemplate
prompt_template = PromptTemplate.from_template(
    "You are my assistant that helps manage my emails. Use the following pieces of retrieved context to accomplish the task you are asked. The context contains emails extracted from Gmail account. Analyse each email document carefully because each detail may be important to me. If you can't run the task, just say that you can't.Be explicit and give details as much as you can. \nTask: {task} \nContext: {context} \nAnswer:"
)

# Injecting our prompts inputs 
final_prompt = prompt_inputs | prompt_template


# Instantiating our AI Brain
from langchain_community.chat_models import ChatOllama
llm = ChatOllama(model="llama3", temperature=0.1)

# Adding the LLM to the chain
rag_llm_chain = final_prompt | llm

# Adding the StrOutputParser to the chain to format the LLM response into a readable text for human
from langchain_core.output_parsers import StrOutputParser
chain = rag_llm_chain | StrOutputParser()

# Invoking our chain 
def generate_response(input_text):
    return chain.invoke(input_text)

        
        
# cleanup
#Svectorstore.delete_collection()


