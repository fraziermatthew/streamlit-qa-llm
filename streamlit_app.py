import os, configparser
import streamlit as st

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain


persist_directory = 'db'
excl_persist_directory = 'excl_db'
embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["openai_api_key"])
model_name = "gpt-3.5-turbo"

st.set_page_config(layout="centered", page_title="AP CSP")
st.header("AP CSP")
st.write("---")

vStore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
excl_vs = Chroma(persist_directory=excl_persist_directory, embedding_function=embeddings)

retriever = vStore.as_retriever()
retriever.search_kwargs = {'k':2}

# Initiate model
llm = OpenAI(model_name=model_name, openai_api_key = st.secrets["openai_api_key"], streaming=True)
model = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

st.header("Chat")
user_q = st.text_area("Enter your text here...")

if st.button("Get Response"):
  try:
    with st.spinner("Model is working on it..."):
      result = model({"question":user_q}, return_only_outputs=True)
      st.subheader('Your response:')
      st.write(result['answer'])
      st.subheader('Source pages:')
      st.write(result['sources'])
  except Exception as e:
    st.error(f"An error occurred: {e}")
    st.error('Oops, the GPT response resulted in an error :( Please try again with a different question.')
