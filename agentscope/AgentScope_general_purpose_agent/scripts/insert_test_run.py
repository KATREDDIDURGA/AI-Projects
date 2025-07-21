# scripts/insert_test_data.py
import asyncio
from datetime import datetime
from app.db.database import async_session, engine
from app.db.base import Base
from app.db.models.agent_run import AgentRun
from app.db.models.agent_step import AgentStep
from app.db.models.agent_metric import AgentMetric
import uuid


async def insert_full_test_trace():
    # Create tables first
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with async_session() as session:
        run_id = "13025c79-065a-42d5-b9d4-915b30442027"  # Use your exact run_id

        # Create agent run
        run = AgentRun(
            run_id=run_id,
            agent_type="refund",
            query="Can I return my keyboard?",
            status="completed",
            started_at=datetime(2025, 7, 21, 10, 0, 0),
            completed_at=datetime(2025, 7, 21, 10, 0, 5),
            final_decision="Approved refund within 30-day policy."
        )

        # Create agent steps
        steps = [
            AgentStep(
                run_id=run_id, 
                step_number=1,
                step_type="thought", 
                description="User is asking for a return. Need to check policy.",
                confidence=0.8
            ),
            AgentStep(
                run_id=run_id, 
                step_number=2,
                step_type="action", 
                description="Checking return policy for product ID.",
                observation="Found product in system",
                confidence=0.9
            ),
            AgentStep(
                run_id=run_id, 
                step_number=3,
                step_type="decision", 
                description="Policy allows returns within 30 days.",
                confidence=0.95
            ),
            AgentStep(
                run_id=run_id, 
                step_number=4,
                step_type="observation", 
                description="Refund approved. Notified user.",
                confidence=1.0
            )
        ]

        # Create some metrics
        metrics = [
            AgentMetric(run_id=run_id, metric_name="confidence_score", value=0.9),
            AgentMetric(run_id=run_id, metric_name="execution_time", value=5.2)
        ]

        session.add(run)
        session.add_all(steps)
        session.add_all(metrics)
        await session.commit()
        print(f"✅ Inserted test run with ID: {run_id}")


if __name__ == "__main__":
    asyncio.run(insert_full_test_trace())