import os
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.vectorstores import Weaviate
import weaviate
from weaviate.embedded import EmbeddedOptions


load_dotenv()


url = "https://raw.githubusercontent.com/langchain-ai/langchain/master/docs/docs/modules/state_of_the_union.txt"
res = requests.get(url)
with open("state_of_the_union.txt", "w") as f:
    f.write(res.text)


loader = TextLoader("state_of_the_union.txt")
docs = loader.load()

text_splitter = CharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64,
)

chunks = text_splitter.split_documents(docs)

# print(chunks, len(chunks))


## weaviate
client = weaviate.Client( # weaviate from package
    embedded_options=EmbeddedOptions()
)

vectorstore = Weaviate.from_documents( # weaviate from langchain
    client,
    documents = chunks,
    embeddings = OpenAIEmbeddings(),
    by_text = False
) 


# RETRIEVE
retriever = vectorstore.as_retriever()

# AUGMENT
template = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)
print(prompt)

# GENERATE
llm = ChatOpenAI(
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

chain = (
    {"retriever": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

query = st.text_input('Ask question about bidens 2020 speech...')

if query:
    response = chain.invoke({"question": query})
    st.write(response)