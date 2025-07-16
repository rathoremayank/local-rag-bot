# Local RAG Bot using 

## How to  run

1. Start Ollama

```
ollama serve
```
2. Run Ollama (optional for testing)
```
ollama run llama3
```
3. Generate faiss index

```
python3 generate_index.py
```

4. Start the python app

```
uvicorn app.main:app --reload
```
