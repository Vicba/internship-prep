import os
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI
from dotenv import load_dotenv
from langchain.callbacks import get_openai_callback
import langchain

langchain.verbose = False # When set to True, it provides a more detailed, step-by-step output. 

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

def process_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        length_function=len,
    )

    chunks = text_splitter.split_text(text)

    embeddings = OpenAIEmbeddings()

    vectorstore = FAISS.from_texts(chunks, embeddings)

    return vectorstore




def main():
    st.title("Chat with my PDF")

    pdf = st.file_uploader("Upload your PDF File", type="pdf")

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        text = ""

        for page in pdf_reader.pages:
            text += page.extract_text()

        print(text)
        vectorstore = process_text(text)


        query = st.text_input('Ask question to PDF...')
        cancel_button = st.button('Cancel')

        if cancel_button:
            st.stop()

        if query:
            docs = vectorstore.similarity_search(query, k=2)

            llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

            chain = load_qa_chain(llm, chain_type="stuff")


            with get_openai_callback() as cost:
                response = chain.invoke(input={"question": query, "input_documents": docs})
                print(cost)

            st.write(response["output_text"])
            st.write(cost)





if __name__ == "__main__":
  main()