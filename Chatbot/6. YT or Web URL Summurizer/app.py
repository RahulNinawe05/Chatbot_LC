import validators,streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
import os
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="Langchian: Summarize Text From YT OR Website", 
                   page_icon='🦜',
                   layout='wide'
)
st.title("🦜 Langchian: Summarize Text From YT OR Website")
st.subheader('Summarize URL')


with st.sidebar:
    groq_api_key=st.text_input('GROQ_API_KEY', value="", type='password')

llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key)

prompt_tmplate= (
"""provided the summury of following content:
summary:{text}
"""
)
prompt = PromptTemplate(
    template=prompt_tmplate,
    input_variables=["text"]
)

genral_url = st.text_input("URL", label_visibility='collapsed')

if st.button('Summarize'):
    if not groq_api_key.strip() or not genral_url.strip():
        st.error("Please Provided The Information")
    elif not validators.url(genral_url):
        st.error("Please Provided valid Information It can may be a YT or Website URL")
    else:
        try:
            with st.spinner("waiting..."):
                # loading
                if "youtube.com" in genral_url:
                    loader = YoutubeLoader.from_youtube_url(genral_url,add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(urls=[genral_url],verify=False,
                                                   headers={ "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)","Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",})

                docs = loader.load()

                # chain for summarization
                chain = load_summarize_chain(llm=llm, chain_type="stuff", prompt=prompt)
                output_summary=chain.run(docs)

                st.success(output_summary)
        except Exception as e:
            st.error(f"Exception: {e}")