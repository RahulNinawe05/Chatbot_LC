from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os

from dotenv import load_dotenv
load_dotenv()

## Lansgsmith Tracking
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
if langchain_api_key:
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
else:
    st.warning("LANGCHAIN_API_KEY not found in .env file. Please set it.")

os.environ['LANGCHAIN_TRACING_V2']="true"
os.environ['LANGCHAIN_PROJECT']= "Simple Q&A Chatbot With Groq"

# prompts
prompts = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user queries."),
    ("user", "Question: {queries}")
])

def genrate_responce(question,engine,temperature,max_tokens):
    llm = Ollama(model=engine)
    OutputParser = StrOutputParser()
    chain = prompts | llm | OutputParser
    answer= chain.invoke({'question':question})
    return answer

# engine
engine = st.sidebar.selectbox(
    "Select Ollma MOdel",
    ["Gemma 3","Gemma 3","Llama 3.3"]
)

# Temperature
temperature=st.sidebar.slider("Temprature", min_value=0.0,max_value=1.0,value=0.5)

# max_tokens
max_tokens = st.sidebar.slider("Max Tokens", min_value=50,max_value=150,value=100)


# Main Interface for user input
st.write("Get ahead ask Your question")
user_input=st.text_input("You: ")

if user_input:
    responce = genrate_responce(user_input,engine,temperature,max_tokens)
    st.write(responce)
else:
    st.write("Please Provided User Input")
