from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
import shutil
from tempfile import NamedTemporaryFile


def extract_pdf_text(uploaded_file):

    with open(uploaded_file.name, mode='wb') as w:
        w.write(uploaded_file.getvalue())

    loader = PyPDFLoader(uploaded_file.name)
    content = loader.load()
    return content


db_location = "./chroma_langchain_db"
add_documents = not os.path.exists(db_location)

vector_store = Chroma(
    collection_name="pdf_data",
    embedding_function=OllamaEmbeddings(model = "mxbai-embed-large"),
    persist_directory=db_location
)


def split_text(documents: list[Document]):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        length_function = len,
        is_separator_regex=False

        )
    chunks = text_splitter.split_documents(documents)
    
    return chunks


def add_to_db(documents: list[Document]):
    if add_documents:
        vector_store.add_documents(documents)
    



retriever = vector_store.as_retriever(
   search_kwargs = {"k":5}
)







