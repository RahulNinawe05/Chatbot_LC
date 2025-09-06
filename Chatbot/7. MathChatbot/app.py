import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.chains.llm_math.base import LLMMathChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv

# set up st
st.set_page_config(page_title="Text To Math Chatbot", page_icon="🧮")
st.title("Text To Math Problem Solver Using Google Gemma 2")

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.info("Please Provided Groq Api Key")
    st.stop()

llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

## Initializing the tools
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool=Tool(
    name='Wikipedia',
    func=wikipedia_wrapper.run,
    description="The tool for searchaing in Internel to find various information on the topics Mentioned"
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


def clear_history():
   st.session_state["messages"]=[
        {"role":"assistant", "content":"Hi, I'm A Math Chatbot Who Can Answer All Your Math Question"}
    ]     

if "messages" not in st.session_state:
    clear_history()

st.sidebar.button("Clear History",on_click=clear_history)


# combine all tool in the chain
llm_chain=LLMChain(llm=llm,prompt=Prompt_Template)

reasoning_tool=Tool(
    name="Reasoning tool",
    func=llm_chain.run,
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


for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])


# lets start the intraction
with st.container():
    user_input = st.text_area("Type Your Question",key="input_box")
    ask_button = st.button("Ask")


if ask_button and user_input:
    st.session_state["messages"].append({"role":"user", "content":user_input})
    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container())
        responce = agent.invoke({"input":user_input},{"callbacks":[st_cb]})
        final_answer = responce["output"]
        st.write(final_answer)

    # save assistant responce 
    st.session_state['messages'].append({"role":"assistant", "content": final_answer})

