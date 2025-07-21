from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from app.execution.execution_service import ExecutionService

router = APIRouter()
execution_service = ExecutionService()


class RunAgentRequest(BaseModel):
    agent: str
    query: str
    metadata: Optional[dict] = None


class Step(BaseModel):
    step_number: int
    step_type: str
    description: str
    observation: Optional[str]
    confidence: Optional[float]


class RunAgentResponse(BaseModel):
    run_id: str
    final_result: str
    confidence: Optional[float]
    steps: List[Step]
    fallback_reason: Optional[str] = None


@router.post("/run-agent", response_model=RunAgentResponse)
async def run_agent_endpoint(payload: RunAgentRequest):
    try:
        result = await execution_service.run_agent(
            agent_name=payload.agent,
            query=payload.query,
            metadata=payload.metadata
        )

        return RunAgentResponse(
            run_id=result.run_id,
            final_result=result.final_result,
            confidence=result.confidence,
            steps=result.steps,
            fallback_reason=result.fallback_reason,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
