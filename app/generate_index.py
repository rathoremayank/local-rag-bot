# generate_index.py
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Customize this to your document path
loader = TextLoader("../docs/knowledge-fruits.txt")
documents = loader.load()

# Split into chunks
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = splitter.split_documents(documents)

# Load embedding model
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Build FAISS vector DB
db = FAISS.from_documents(docs, embedding)

# Save it to 'faiss_index'
db.save_local("../faiss_index")

print("✅ FAISS index created and saved to ./faiss_index")
