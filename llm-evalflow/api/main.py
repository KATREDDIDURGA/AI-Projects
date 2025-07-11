# api/main.py
from fastapi import FastAPI
import json
from pathlib import Path

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "LLM EvalFlow API is running."}

@app.get("/results")
def get_results():
    data_path = Path("data/feedback.json")
    if data_path.exists():
        return json.loads(data_path.read_text())
    return {"error": "No feedback data found."}
