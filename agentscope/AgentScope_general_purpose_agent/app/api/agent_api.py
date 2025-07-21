# app/api/agent_api.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.agent_executor import AgentExecutor

router = APIRouter()


class AgentRequest(BaseModel):
    agent_type: str
    query: str


class AgentResponse(BaseModel):
    run_id: str
    agent_type: str
    query: str
    final_decision: str
    error: str | None = None


executor = AgentExecutor()


@router.post("/agent/run", response_model=AgentResponse)
async def run_agent(request: AgentRequest):
    try:
        result = await executor.execute_agent(request.agent_type, request.query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
