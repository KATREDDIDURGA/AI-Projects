# app/services/agent_executor.py

from app.agents.agent_registry import AgentRegistry
from app.db.run_logger import RunLogger
from app.db.database import async_session
from typing import Dict, Any


class AgentExecutor:
    def __init__(self):
        self.registry = AgentRegistry()

    async def execute_agent(self, agent_type: str, query: str) -> Dict[str, Any]:
        agent = self.registry.get_agent(agent_type)

        async with async_session() as session:
            logger = RunLogger(session)

            # Start new agent run
            run_id = await logger.start_run(agent_type, query)
            agent.set_context(run_id=run_id, logger=logger)

            try:
                # Execute core logic
                result = await agent.run(query)
                final_decision = result.get("final_decision", "No decision made")

                # Complete run
                await logger.complete_run(run_id, final_decision)

                return {
                    "run_id": run_id,
                    "agent_type": agent_type,
                    "query": query,
                    "final_decision": final_decision
                }

            except Exception as e:
                await logger.complete_run(run_id, f"Failed: {str(e)}")
                return {
                    "run_id": run_id,
                    "agent_type": agent_type,
                    "query": query,
                    "final_decision": "Error",
                    "error": str(e)
                }
