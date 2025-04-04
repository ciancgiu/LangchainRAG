from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
import shutil

pdf_path = "dsa.pdf"
loader = PyPDFLoader(pdf_path)
content = loader.load()
embeddings = OllamaEmbeddings(model = "mxbai-embed-large")

db_location = "./chroma_langchain_db"
add_documents = not os.path.exists(db_location)

text_splitter = RecursiveCharacterTextSplitter(
    chunks_size=800,
    overlap=80,
    length_function = len,
    is_separator_regex=False
)
chunks = text_splitter.split_text(content)

if add_documents:
    documents = []

    for i, chunk in enumerate(chunks):
            document = Document(
                page_content=chunk,
                metadata={"chunk": i},
            )    
    documents.append(document)

vector_store = Chroma(
    collection_name = "pdf_chunks",
    persist_directory=db_location,
    embedding_function=embeddings
)
if add_documents:
    vector_store.add_documents(documents=documents)

retriever = vector_store.as_retriever(
    search_kwargs = {"k":5}
)

def clearDB():
    shutil.rmtree(db_location)





