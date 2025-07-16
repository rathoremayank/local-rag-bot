from fastapi import FastAPI
from pydantic import BaseModel
from app.rag_engine import get_qa_chain

app = FastAPI()
qa_chain = get_qa_chain()

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask(query: Query):
    result = qa_chain.run(query.question)
    return {"answer": result}
