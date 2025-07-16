from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from app.loader import load_documents, split_documents
from app.config import EMBED_MODEL, INDEX_PATH, DOCS_PATH
import os

# Load or build vectorstore
def get_vectorstore():
    if os.path.exists(INDEX_PATH):
        db = FAISS.load_local(
            INDEX_PATH,
            HuggingFaceEmbeddings(model_name=EMBED_MODEL),
            allow_dangerous_deserialization=True  # ✅ This line is required now
            )
    else:
        docs = split_documents(load_documents(DOCS_PATH))
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        db = FAISS.from_documents(docs, embeddings)
        db.save_local(INDEX_PATH)
    return db

# Build RAG chain
def get_qa_chain():
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever()
    llm = Ollama(model="llama3")  # Assumes Ollama is running
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
