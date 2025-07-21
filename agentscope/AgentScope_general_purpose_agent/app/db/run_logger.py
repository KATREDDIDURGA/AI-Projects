# app/db/run_logger.py

from sqlalchemy.ext.asyncio import AsyncSession
from app.models.agent_run import AgentRun
from app.models.agent_step import AgentStep
from app.models.agent_metric import AgentMetric
from datetime import datetime
import uuid


class RunLogger:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def start_run(self, agent_type: str, query: str) -> str:
        run_id = f"{agent_type}-{uuid.uuid4().hex[:8]}"
        run = AgentRun(
            run_id=run_id,
            agent_type=agent_type,
            query=query,
            status="started",
            started_at=datetime.utcnow()
        )
        self.session.add(run)
        await self.session.commit()
        return run_id

    async def complete_run(self, run_id: str, final_decision: str):
        run = await self.session.get(AgentRun, run_id)
        if run:
            run.status = "completed"
            run.completed_at = datetime.utcnow()
            run.final_decision = final_decision
            await self.session.commit()

    async def log_step(self, run_id: str, step_number: int, thought: str, observation: str, action: str, result: str, severity: str = None):
        step = AgentStep(
            run_id=run_id,
            step_number=step_number,
            thought=thought,
            observation=observation,
            action=action,
            result=result,
            severity=severity,
        )
        self.session.add(step)
        await self.session.commit()

    async def log_metric(self, run_id: str, key: str, value: str):
        metric = AgentMetric(
            run_id=run_id,
            metric_key=key,
            metric_value=value
        )
        self.session.add(metric)
        await self.session.commit()
