from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
import shutil

pdf_path = "dsa.pdf"
loader = PyPDFLoader(pdf_path)
content = loader.load_and_split
embeddings = OllamaEmbeddings(model = "mxbai-embed-large")

db_location = "./chroma_langchain_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []

    for i, page in enumerate(content):
        document = Document(
            document = Document(
                page_content=page,
                metadata={"page_number": i+1},
                id = str(i)
            )     
        )
        ids.append(str(i))
        documents.append(document)

vector_store = Chroma(
    collection_name = "pdf_pages",
    persist_directory=db_location,
    embedding_function=embeddings
)
if add_documents:
    vector_store.add_documents(documents=documents,ids = ids)

retriever = vector_store.as_retriever(
    search_kwargs = {"k":5}
)

def clearDB():
    shutil.rmtree(db_location)





