from fastapi import FastAPI, Request
from pydantic import BaseModel
from agent import run_refund_fraud_agent_polling, agent_runs
import uuid

app = FastAPI()

class UserQuery(BaseModel):
    query: str

@app.post("/init-agent-run/")
async def init_agent_run(query: UserQuery):
    run_id = str(uuid.uuid4())
    agent_runs[run_id] = {
        "steps": [],
        "query": query.query,
        "done": False,
        "final": None
    }
    run_refund_fraud_agent_polling(run_id, query.query)
    return {"run_id": run_id}

@app.get("/get-next-step/{run_id}")
async def get_next_step(run_id: str):
    run = agent_runs.get(run_id)
    if not run:
        return {"error": "Invalid run_id"}
    return {
        "steps": run["steps"],
        "done": run["done"],
        "final": run["final"]
    }