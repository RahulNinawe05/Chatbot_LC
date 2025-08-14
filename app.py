import os
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load env
load_dotenv()

# LangSmith tracking
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_PROJECT'] = "Chatbot_ST_LC"

# Prompt
prompts = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user queries."),
    ("user", "Question: {queries}")
])

# temperature :- 
# Low temperature (0–0.3) → More predictable, accurate, and less random answers.
# Medium (0.5–0.7) → Balanced — some creativity, some accuracy.
# High (0.8–1.0+) → More creative, random, and varied responses, but sometimes less accurate.

def generate_responce(question, api_key, llm, temperature, max_tokens):
    api_key = api_key  
    llm = ChatGroq(model=llm) 
    output_parser = StrOutputParser()  

    chain = prompts | llm | output_parser
    answer = chain.invoke({'queries': question})
    return answer

## Title of the app
st.title("Enhanced Q&A Chatbot With Groq")

## sidebar for setting
st.sidebar.title("Setting")
api_key= st.sidebar.text_input("Enter Your Groq Api Key", type="password")

## Drop down to select various Groq Models
llm = st.sidebar.selectbox("Select an Groq Model", ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "whisper-large-v3", "whisper-large-v3-turbo"])

# Adjust response parameter

temperature=st.sidebar.slider("Temperature",min_value=0.0, max_value=1.0,value=0.7)
max_tokens=st.sidebar.slider("Max_tokens",min_value=50,max_value=300,value=150)

## Main inteface for user input

st.write("Give me Question")
user_input = st.text_input("You: ")

if user_input:
    responce = generate_responce(user_input,api_key,llm,temperature,max_tokens)
    st.write(responce)
else:
    st.write("Please provided the Query")