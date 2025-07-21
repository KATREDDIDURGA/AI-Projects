'''
💡 What This Is:
The core engine that:

Loads the correct agent

Executes logic step-by-step

Tracks fallback triggers

Logs intermediate reasoning

Streams data to Web UI via WebSocket

🧠 Why We Need This:
This is the brain that ensures traceability

Enables live debugging for devs

Manages concurrency, cancellation, observability

Records decisions, fallbacks, and step logs

⏰ When It’s Used:
Every time a query is submitted to an agent.

'''
import uuid
import datetime
from typing import Callable, Optional

from app.agents.base_agent import AgentContext, BaseAgent
from app.agents.agent_registry import AgentRegistry
from app.agents.fallbacks import FallbackTrigger


class AgentExecutionResult:
    def __init__(self, run_id, steps, final_result, fallback_reason=None, confidence=None):
        self.run_id = run_id
        self.steps = steps
        self.final_result = final_result
        self.fallback_reason = fallback_reason
        self.confidence = confidence
        self.timestamp = datetime.datetime.utcnow()


class ExecutionService:
    def __init__(self):
        self.registry = AgentRegistry()

    async def run_agent(
        self,
        agent_name: str,
        query: str,
        metadata: Optional[dict] = None,
        on_step: Optional[Callable] = None,
    ) -> AgentExecutionResult:
        agent: BaseAgent = self.registry.get_agent(agent_name)
        run_id = str(uuid.uuid4())
        context = AgentContext(run_id=run_id, query=query, metadata=metadata or {})
        steps = []

        async def log_step(ctx, step_type, description, observation=None, confidence=None):
            step = {
                "step_number": ctx.current_step,
                "step_type": step_type,
                "description": description,
                "observation": observation,
                "confidence": confidence,
            }
            steps.append(step)
            if on_step:
                await on_step(step)

        agent._log_step = log_step

        try:
            final_result = await agent.execute_logic(context)
            return AgentExecutionResult(run_id, steps, final_result, None, context.overall_confidence)

        except FallbackTrigger as fb:
            return AgentExecutionResult(run_id, steps, f"FALLBACK: {fb.reason}", fb.reason, fb.confidence)
