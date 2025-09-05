import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.chains.llm_math.base import LLMMathChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler

# set up st
st.set_page_config(page_title="Text To Math Chatbot", page_icon="🦜")
st.title("Text To Math Problem Solver Using Google Gemma 2")

groq_api_key = st.sidebar.text_input(label="Groq API Key", type="password")

if not groq_api_key:
    st.info("Please Provided Groq Api Key")
    st.stop()

llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

## Initializing the tools
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool=Tool(
    name='Wikipedia',
    func=wikipedia_wrapper.run,
    description="The tool for searchaing in Interner to find various information on the topics Mentioned"
)

# Initializing The math tool
math_chain = LLMMathChain.from_llm(llm=llm)
calculate_tool = Tool(
    name='Calculator',
    func=math_chain.run,
    description="The tool answering the math related Question, Only input mathematical expression needs to be Provided"
)

prompt = """
Your a agent tasked for solving users mathematical question. logically arrive at the solution and provide the detaild explenation
and desplay it point wise for the question below
Question:{question}
Answer:
"""

Prompt_Template=PromptTemplate(
    input_variables=["question"],
    template=prompt
)

# combine all tool in the chain
llm_chain=LLMChain(llm=llm,prompt=Prompt_Template)

reasoning_tool=Tool(
    name="Reasoning tool",
    func=llm_chain.run(),
    description="A tool for answering logic-based and reasoning question."
)

## initialize the agents

agent=initialize_agent(
    tools=[wikipedia_tool,calculate_tool,reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assistant", "content":"HI, I'm A MATH CHATBOT WHO CAN ANSWER ALL YOUR MATH QUESTION"}
    ]
