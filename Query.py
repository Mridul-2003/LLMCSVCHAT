import streamlit as st
import os
import pandas as pd
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import CTransformers



st.title("Querying CSVs with LLMs")

uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
if uploaded_file is not None:
    dataframe = pd.read_csv(uploaded_file)
    st.write(dataframe)
    
    temp_csv_path = "temp_uploaded_file.csv"
    dataframe.to_csv(temp_csv_path, index=False)
    
    loader = CSVLoader(file_path=temp_csv_path)
    data = loader.load() 

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(data)
    
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.from_documents(text_chunks,embeddings)
    
    llm = CTransformers(model="/Users/useradmin/Documents/Assignment_TensorGo/models/llama-2-7b-chat.ggmlv3.q4_0.bin"
                        ,model_type="llama",
                        max_new_tokens=512,
                        temperature=0.1)
    
    qa = ConversationalRetrievalChain.from_llm(llm, retriever=db.as_retriever())
    
    st.write("Enter your Query..")
    query = st.text_input("Input Prompt: ")
    
    if query:
        with st.spinner("Processing your question.."):
            chat_history=[]
            result=qa({"question":query,"chat_history":chat_history})
            st.write("Response:", result["answer"])