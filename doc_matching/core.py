import os
from typing import Any

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Pinecone
from langchain_pinecone import PineconeVectorStore
import pinecone

INDEX_NAME="docs-analyser"



def run_llm(query: str) -> Any:
    embeddings = OpenAIEmbeddings()
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME, embedding=embeddings
    )
    chat = ChatOpenAI(verbose=True, temperature=0)
    qa = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
    )
    return qa({"query": query})["result"]


if __name__ == "__main__":
    print(run_llm(query="java engineer in phillipines experienced in springboot:write the name and contact details and skills "))
