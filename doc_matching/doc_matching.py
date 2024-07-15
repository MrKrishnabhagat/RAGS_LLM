from langchain_community.document_loaders import PyPDFLoader,Docx2txtLoader
import numpy as np
import logging 
import os
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter,CharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from pinecone import Pinecone as pc
import os
import logging
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone.vectorstores import PineconeVectorStore
import os
import logging
from pinecone import Pinecone, ServerlessSpec
from langchain_community.vectorstores import Pinecone as LC_Pinecone
from langchain_openai import OpenAIEmbeddings
from typing import List
import pinecone
from langchain.schema import Document

import os
import json

from langchain.schema import Document





from pinecone.core.client.exceptions import PineconeApiException
# def insert_embeddings(index_name, chunks):



#     embeddings = OpenAIEmbeddings()
#     pinecone.init(api_key=os.environ.get("PINECONE_API_KEY"), environment=os.environ.get("PINECONE_ENV"))

#     vector_store = Pinecone(index_name=index_name, embeddings=embeddings)
    
#     # Batch insert the chunks into the vector store
#     batch_size = 700  # Define your preferred batch size
#     for i in range(0, len(chunks), batch_size):
#         chunk_batch = chunks[i:i + batch_size]
#         vector_store.add_documents(chunk_batch)

#     # Flush the vector store to ensure all documents are inserted
#     vector_store.flush()

#     print("Ok")



# class Document:
#     def __init__(self, page_content, metadata):
#         self.page_content = page_content
#         self.metadata = metadata 
def pdf_to_chunks(pdf):
        # read pdf and it returns memory address
    pdf_reader = PdfReader(pdf)

        # extrat text from each page separately
    text = ""
    for page in pdf_reader.pages:
            text += page.extract_text()
    

        # Split the long text into small chunks.
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=200,
            length_function=len)

    chunks = text_splitter.split_text(text=text)
    return chunks

def docx_splitter(path: str):
    # Load the document
    loader = Docx2txtLoader(path)
    
    # Extract text from the document
    documents = loader.load()
    
    # Join the text content of each Document object into a single string
    text = "\n".join(doc.page_content for doc in documents)
  
    # Split the long text into small chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text=text)
    
    return chunks


def embedder():
    # Initialize Pinecone
    pc = Pinecone(api_key="69e46d20-e1e7-4fce-be5d-4f294662a635")
    INDEX_NAME ="docs-analyser"

    # Debugging statement
    print(f"INDEX_NAME: {INDEX_NAME}")

    if INDEX_NAME is None:
        raise ValueError("INDEX_NAME environment variable is not set")

    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,
            metric='euclidean',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )

    files_dir = "/Users/krkd/Desktop/doc_matching/resumes"
    files = [f for f in os.listdir(files_dir) if os.path.isfile(os.path.join(files_dir, f))]

    for file_name in files:
        file_path = os.path.join(files_dir, file_name)
        file_extension = file_path.split('.')[-1]
        logging.info(f"Processing file: {file_path}")

        if file_extension == "docx":
            doc = docx_splitter(file_path)
            insert_embeddings(pc, INDEX_NAME, doc,file_path)
        elif file_extension == "pdf":
            doc = pdf_to_chunks(file_path)
            insert_embeddings(pc, INDEX_NAME, doc,file_path)





from PyPDF2 import PdfFileReader  # Ensure you have PyPDF2 imported
  # Ensure PyCryptodome is installed and can be imported


def insert_embeddings(pc, index_name, doc, file_path):
    text = "".join(doc)
    metadata = {"file_name": os.path.basename(file_path)}

    def truncate_metadata(metadata, limit):
        metadata_json = json.dumps(metadata)
        metadata_size = len(metadata_json.encode('utf-8'))
        
        if metadata_size <= limit:
            return metadata

        for key in metadata:
            while len(json.dumps(metadata).encode('utf-8')) > limit:
                if len(metadata[key]) > 1:
                    metadata[key] = metadata[key][:-1]
                else:
                    break
        return metadata

    METADATA_LIMIT = 40960
    metadata = truncate_metadata(metadata, METADATA_LIMIT)
    
    metadata_size = len(json.dumps(metadata).encode('utf-8'))
    print(f"Metadata size after truncation (if applied): {metadata_size} bytes")

    if metadata_size > METADATA_LIMIT:
        print(f"Skipping document due to oversized metadata (size: {metadata_size} bytes).")
        return

    document = Document(page_content=text, metadata=metadata)
    documents = [document]
    
    embeddings = create_embeddings(doc)
    
    print(f"documents: {documents}")
    print(f"embeddings: {embeddings}")

    try:
        PineconeVectorStore.from_documents(documents=documents, embedding=embeddings, index_name=index_name)
    except PineconeApiException as e:
        print(f"Failed to insert document due to Pinecone API exception: {e}. Skipping document.")
    except RuntimeError as e:
        print(f"Runtime error: {e}. Skipping document.")

def create_embeddings(doc):
    embeddings = OpenAIEmbeddings()
    return embeddings

INDEX_NAME = os.getenv("INDEX_NAME", "default-index")
pc = None  # Initialize your Pinecone client here
doc = ["Sample text from document..."]
file_path = "/path/to/document"

insert_embeddings(pc, INDEX_NAME, doc, file_path)



if __name__ == "__main__":
    embedder()
    print("*******************")

