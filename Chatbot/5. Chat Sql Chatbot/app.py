import streamlit as st
from pathlib import Path
from langchain.agents import create_sql_agent 
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.callbacks import StreamlitCallbackHandler
from sqlalchemy import create_engine
import sqlite3
from langchain_groq import ChatGroq

st.set_page_config(page_title='Langchain: Chat With SQL DB',
                   page_icon='🦜',
                   layout="wide"
)
st.title("🦜 Langchain: Chat With SQL DB")

LOCALDB="USE_LOCALDB"
MYSQL="USE_MYSQL"

radio_opt=["USE SQLLITE 3 DATABASE- class_db","Connected To MYSQL Database"]

selected_opt = st.sidebar.radio(label="Choose The DATABASE",options=radio_opt)

if radio_opt.index(selected_opt)==1:
    db_uri = MYSQL
    mysql_host = st.sidebar.text_input("Provided MySQL Host")
    mysql_user = st.sidebar.text_input("MYSQL User")
    mysql_password=st.sidebar.text_input("MYSQL Password", type="password")
    mysql_db=st.sidebar.text_input("MYSQL DATABASE")
else:
    db_uri=LOCALDB

api_key=st.sidebar.text_input(label="GROQ API KEY", type='password')

if not db_uri:
    st.info("Please Enter The DATABASE Information and URL")

if not api_key:
    st.info("Please Provide The GROQ API KEY")

## LLM MODEL
ChatGroq(GROQ_API_KEY=api_key,model_name="Llama3-8b-8192", streaming=True)

@st.cache_resource(ttl='2h')
def configure_db(db_uri,mysql_host=None,mysql_user=None,mysql_password=None,mysql_db=None):
    if db_uri == LOCALDB:
        db_file_path=(Path(__file__).parent/"class_db.db").absolute()
        print(db_file_path)
        creator = lambda:sqlite3.connect(f"file:{db_file_path}?mode=ro", uri=True)
        return SQLDatabase(create_engine("sqlite:///", creator=creator))
    elif db_uri==MYSQL:
        if not (mysql_host and mysql_user and mysql_password and mysql_db):
            st.error("Please Provided all MYSQL Connection detailes.")
            st.stop()
        return SQLDatabase(create_engine(f"mysql+mysqlconnctor://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"))
    
if db_uri==MYSQL:
    db=configure_db(db_uri,mysql_host,mysql_db,mysql_password,mysql_user)
else:
    db=configure_db(db_uri)

## toolkit