import os
from dotenv import load_dotenv
load_dotenv()

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3")
DOCS_PATH = "docs"
INDEX_PATH = "faiss_index"
