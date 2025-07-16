from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def load_documents(doc_path: str):
    documents = []
    for filename in os.listdir(doc_path):
        path = os.path.join(doc_path, filename)
        if filename.endswith(".pdf"):
            documents += PyMuPDFLoader(path).load()
        elif filename.endswith(".docx"):
            documents += Docx2txtLoader(path).load()
        elif filename.endswith(".txt"):
            documents += TextLoader(path).load()
    return documents

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(documents)
