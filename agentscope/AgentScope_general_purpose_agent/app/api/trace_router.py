"""
🔍 Trace Router — Returns agent run logs (for UI/debugging)

💡 What:
    FastAPI route to fetch a full trace (AgentRun, AgentSteps, AgentMetrics) by run_id.

🧠 Why:
    Powers the frontend and lets developers debug step-by-step what the agent did.

🕐 When:
    Called from the UI (or Postman) with a Run ID → /trace/{run_id}

🔧 How:
    Async DB fetch using SQLAlchemy ORM, joins all tables by run_id.
"""

from fastapi import APIRouter, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.database import async_session
from app.models.agent_run import AgentRun
from app.models.agent_step import AgentStep
from app.models.agent_metric import AgentMetric

router = APIRouter()

@router.get("/trace/{run_id}")
async def get_agent_trace(run_id: str):
    async with async_session() as session:
        # Fetch the agent run
        run_result = await session.execute(
            select(AgentRun).where(AgentRun.run_id == run_id)
        )
        run = run_result.scalar_one_or_none()
        if not run:
            raise HTTPException(status_code=404, detail="Run ID not found")

        # Fetch steps
        step_result = await session.execute(
            select(AgentStep).where(AgentStep.run_id == run_id).order_by(AgentStep.timestamp)
        )
        steps = step_result.scalars().all()

        # Fetch metrics
        metric_result = await session.execute(
            select(AgentMetric).where(AgentMetric.run_id == run_id)
        )
        metrics = metric_result.scalars().all()

        return {
            "run": run.to_dict(),
            "steps": [s.to_dict() for s in steps],
            "metrics": [m.to_dict() for m in metrics]
        }
