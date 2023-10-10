import os, configparser
import streamlit as st

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.chains import LLMChain, RetrievalQA

persist_directory = 'db'
excl_persist_directory = 'excl_db'
embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["openai_api_key"])
model_name = "gpt-3.5-turbo"

st.set_page_config(layout="centered", page_title="AP CSP")
st.header("AP CSP")
st.write("---")

vStore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
excl_vs = Chroma(persist_directory=excl_persist_directory, embedding_function=embeddings)

_template = """You are a tutor for a high school student Computer Science Principles course.
            Given the following documents and a question, create an answer the question.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            All answers must be understandable for a high school student.
            All answers should be succinct.
            Encourage the learner to reflect on their personal experience by using follow-up questions.
            Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

search_kwargs = {"k": 1}
chatgpt = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")
memory = ConversationSummaryMemory(llm=chatgpt,memory_key="chat_history", return_messages=True)
retriever = vs.as_retriever(search_kwargs=search_kwargs)
model = ConversationalRetrievalChain.from_llm(llm=chatgpt,
                                             retriever=vsr,
                                             memory=memory,
                                             verbose=True,
                                             condense_question_prompt=CONDENSE_QUESTION_PROMPT
                                             )

excl_prompt_template = """Use the following pieces of context to reformulate the text. Reformulate your answer to a less technical understanding for a high school student if it should have been excluded.

{context}

Text: {question}
Simplified Text:"""
_excl_template = PromptTemplate(template=excl_prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": _excl_template}
ex_search_kwargs = {"k": 1,"score_threshold": 0.85}
excl_chatgpt=OpenAI(temperature=0, model="gpt-3.5-turbo-instruct")
excl_retriever=excl_vs.as_retriever(search_type="similarity_score_threshold", search_kwargs=ex_search_kwargs)
excl_model = RetrievalQA.from_chain_type(llm=excl_chatgpt,
                                         chain_type="stuff",
                                         retriever=excl_retriever,
                                         chain_type_kwargs=chain_type_kwargs,
                                         verbose=True)

st.header("Chat")
user_q = st.text_area("Enter your text here...")

if st.button("Get Response"):
  try:
    with st.spinner("Model is working on it..."):
      result = model({"question":user_q}, return_only_outputs=True)
      excl_result = excl_model({"question":result['answer']}, return_only_outputs=True)
      # TODO need if condiditon - if past threshold send excl_result else send result
      st.subheader('Your response:')
      st.write(excl_result['answer'])
  except Exception as e:
    st.error(f"An error occurred: {e}")
    st.error('Oops, the GPT response resulted in an error :( Please try again with a different question.')
