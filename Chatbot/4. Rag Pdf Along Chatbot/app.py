## Rag Q&A chatbot with PDF includeing chat history

import streamlit as st
# Retriever + History-aware retriever → find relevant chunks
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain 
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory 
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.chains import create_history_aware_retriever
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

import os
from dotenv import load_dotenv
load_dotenv()

# embedding 
os.environ['HF_TOKEN']= os.getenv("HF_TOKEN") 
Embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# set page configure
st.set_page_config(page_title="Conversation Rag Chatbot",
                   page_icon="🦜",
)
# set up  streamlit  app
st.title("Conversationl Rag With PDF Uploads And Chat History")
st.write("Upload Pdf's File's")

# input the Groq Api Key
api_key = st.text_input("Enter Your Groq Api Key:",type="password")

## if check groq api key is provided or not 
if api_key:
    llm = ChatGroq(groq_api_key=api_key,model_name="llama-3.1-8b-instant")

    # chat Interface
    session_id=st.text_input("SESSION_ID",value='Defoult_session')

    # Statafully manage chat history

    if 'store' not in st.session_state:
        st.session_state.store={}
    
    uploaded_files=st.file_uploader("Choose a pdf file", type="pdf", accept_multiple_files=True)

    folder = r"E:\1-Q&A Chatbot\Chatbot\4. Rag Pdf Along Chatbot"
    
    # process uploaded pdf's
    if uploaded_files:
        documents=[]
        for uploaded_file in uploaded_files:
            temp_pdf=f"{folder}\\temp.pdf"
            with open(temp_pdf, "wb") as file:
                file.write(uploaded_file.getvalue())
                file_name=uploaded_file.name

            loader=PyPDFLoader(temp_pdf)
            docs=loader.load()
            documents.extend(docs)
        
        # split and create embedding form given document
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
        splits=text_splitter.split_documents(documents)
        VectoreStore=FAISS.from_documents(documents=splits,embedding=Embedding)
        retriever=VectoreStore.as_retriever()

        contextualize_q_system_prompt=(
            "Given a chat history and the letest user question"
            "which might refrence  context in the chat history"
            "farmulate  a standalone question which can be understand"
            "without the chat history . Do not answer the question"
            "just reformulate it if needed and othrewise return it as is"
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )


        history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

        ## Answer question prompts
        system_prompt=(
            "You are an assinstant for question-answering task."
            "use the following pieces of retriever context to answer"
            "the question if you don't know the answer, say that you "
            "don't know. use three sentence maximum and keep the "
            "answer concise"
            "\n\n"
            "{context}"
        )

        qa_prompt= ChatPromptTemplate.from_messages(
            [
                ('system', system_prompt),
                MessagesPlaceholder("chat_history"),
                ('human', "{input}"),
            ]
        )

        # Saare documents ko ek saath jod kar (concatenate karke) ek hi prompt me daal dena aur AI ko dena.
        question_answer_chain = create_stuff_documents_chain(llm,qa_prompt)

        # (jo documents laata hai) aur document chain (jo answer generate karta hai) ko ek pipeline (chain) me jodta hai.
        rag_chain = create_retrieval_chain(history_aware_retriever,question_answer_chain) 

        # BaseChatMessageHistory- Har ek user ke liye ya har ek session ke liye alag chat history maintain karne ke liye use hota hai.
        def get_session_history(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]
        

        # RunnableWithMessageHistory Ye ek wrapper (cover) hai jo rag_chain ko memory deta hai.
        # Matlab: Ab chain past conversation ko yaad rakhega.
        conversational_rag_chain = RunnableWithMessageHistory(  
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )


        user_input= st.text_input("Your Question ")
        if user_input:
            session_history=get_session_history(session_id)
            responce = conversational_rag_chain.invoke(
                {"input":user_input},
                config={
                    "configurable": {"session_id":session_id}
                }, # constructure a key 'abc123' in store
            )
            st.write(st.session_state.store)
            st.write("Assistant:", responce['answer'])
            st.write("chat  history:", session_history.messages)
else:
    st.warning("Please enter the groq api key")