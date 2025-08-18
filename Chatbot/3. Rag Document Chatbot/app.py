import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

import os
from dotenv import load_dotenv
load_dotenv()

# load haggingface
os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")

# creat a Embeddign by huggingface
from langchain_huggingface import HuggingFaceEmbeddings
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# loas the Groq 
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model="llama-3.1-8b-instant"  
)

prompts = ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context only.
    please provided the most accurate responce based on the question
    <context>
    {context}
    <context>
    Question:{input}
    """
)

# read the document
def create_vectore_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = embedding  

        st.session_state.loader = PyPDFDirectoryLoader(r"E:\1-Q&A Chatbot\Chatbot\3. Rag Document Chatbot\Reserch_paper")
        st.session_state.docs = st.session_state.loader.load()

        if not st.session_state.docs:
            st.error("No documents found in 'Reserch_paper' folder!")
            return

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        st.session_state.final_document = st.session_state.text_splitter.split_documents(
            st.session_state.docs[:50]
        )

        if not st.session_state.final_document:
            st.error("Document splitting returned no text chunks!")
            return

        # Debug info
        st.write(f"Loaded {len(st.session_state.docs)} documents")
        st.write(f"Created {len(st.session_state.final_document)} chunks")
        st.write("First chunk preview:", st.session_state.final_document[0].page_content[:200])

        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_document,
            st.session_state.embeddings
        )

user_prompt = st.text_input("Enter Your Query from the resurch paper")

if st.button("Document Embedding"):
    create_vectore_embedding()
    st.write("vector Database is Ready")

import time

if user_prompt:
    document_chian= create_stuff_documents_chain(llm,prompts)
    retriever = st.session_state.vectors.as_retriever()
    retriever_chain=create_retrieval_chain(retriever,document_chian)

    start=time.process_time()
    responce=retriever_chain.invoke({'input':user_prompt})
    print(f"Responce time :{time.process_time()-start}")

    st.write(responce['answer'])

    ## expande the context
    with st.expander("Document similarity search"):
        for i,doc in enumerate(responce['context']):
            st.write(doc.page_content)
            st.write('-------------------------------')