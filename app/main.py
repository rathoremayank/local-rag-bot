from fastapi import FastAPI
from pydantic import BaseModel
from app.rag_engine import get_qa_chain
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
qa_chain = get_qa_chain()

# Add this block
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify ["http://localhost:5500"] etc.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Query(BaseModel):
    question: str

@app.post("/ask")
def ask(query: Query):
    result = qa_chain.run(query.question)
    return {"answer": result}
