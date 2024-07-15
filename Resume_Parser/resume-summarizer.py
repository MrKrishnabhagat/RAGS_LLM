from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
import numpy as np
import logging 
import os
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter

from langchain_community.vectorstores import Pinecone as LC_Pinecone
from langchain_openai import OpenAIEmbeddings
from typing import List

from langchain.schema import Document
import docx2txt

from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
import textwrap
import streamlit as st

load_dotenv()

def process_pdf(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_text(text=text)
    text=""
    for chunk in chunks:
        text+=chunk

    documents = [Document(page_content=text)]
    return documents

def process_docx(path: str):
    loader = Docx2txtLoader(path)
    documents = loader.load()
    text = "\n".join(doc.page_content for doc in documents)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=200,
        length_function=len
    )
    
    chunks = text_splitter.split_text(text=text)
    text=""
    for chunk in chunks:
        text+=chunk

    documents = [Document(page_content=text)]
    return documents

def main():
    st.title("CV data retriever")

    uploaded_file = st.file_uploader("Select CV", type=["docx", "pdf"])

    documents = []
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1]

        st.write("File Details:")
        st.write(f"File Name: {uploaded_file.name}")
        st.write(f"File Type: {file_extension}")

        if file_extension == "docx":
            documents = process_docx(uploaded_file.name)
        elif file_extension == "pdf":
            documents = process_pdf(uploaded_file.name)
        else:
            st.error("Unsupported file format. Please upload a .docx or .pdf file.")
            return
    else:
        file_path = st.text_input("Enter file path:")
        if not file_path:
            st.warning("Please enter a file path.")
            return
        file_extension = file_path.split('.')[-1].lower()

        st.write("File Details:")
        st.write(f"File Path: {file_path}")
        st.write(f"File Type: {file_extension}")

        if file_extension == "docx":
            documents = process_docx(file_path)
        elif file_extension == "pdf":
            documents = process_pdf(file_path)
        else:
            st.error("Unsupported file format. Please provide a valid .docx or .pdf file path.")
            return

    llm = OpenAI(temperature=0)
    prompt_template = """You have been given a Resume to analyse. 
    Write a verbose detail of the following: 
    {text}
    Details:"""
    prompt = PromptTemplate.from_template(prompt_template)

    refine_template = (
        "Your job is to produce a final outcome\n"
        "We have provided an existing detail: {existing_answer}\n"
        "We want a refined version of the existing detail based on initial details below\n"
        "------------\n"
        "{text}\n"
        "------------\n"
        
        "Given the new context, refine the original summary in the following manner:"
        "Name: \n"
        "Email: \n"
        "Key Skills: \n"
        "Experience Summary: \n"
        
       
    )
    refine_prompt = PromptTemplate.from_template(refine_template)
    chain = load_summarize_chain(
        llm=llm,
        chain_type="refine",
        question_prompt=prompt,
        refine_prompt=refine_prompt,
        return_intermediate_steps=True,
        input_key="input_documents",
        output_key="output_text",
    )
    
    result = chain({"input_documents": documents}, return_only_outputs=True)

    st.write("Resume Summary:")
    st.text_area("Text", result['output_text'], height=400)

if __name__ == "__main__":
    main()
