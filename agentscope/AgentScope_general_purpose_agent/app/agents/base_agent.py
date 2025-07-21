''' 💡 What This Is:
A reusable abstract agent class that all domain-specific agents (fraud, legal, support) will inherit.

🧠 Why We Need This:
Centralizes core logic (step logging, fallback handling, LLM calls)

Forces structure: every agent will have execute_logic() + fallback patterns

Promotes extensibility without tight coupling

⏰ When It Runs:
Each time an agent is executed — this is the brain of step tracking and fallbacks.

🛠️ How It Works:
BaseAgent is abstract (can’t be instantiated)

Implements: step logging, fallback trigger, LLM usage via llm_client

Subclasses only need to override execute_logic(context)'''
import abc
from typing import Optional, Dict, Any, List
from app.agents.agent_context import AgentContext
from app.llm.llm_client import LLMClient


class StepType:
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    DECISION = "decision"


class FallbackTrigger(Exception):
    def __init__(self, reason: str, suggestion: Optional[str] = None, confidence: float = 0.0):
        self.reason = reason
        self.suggestion = suggestion
        self.confidence = confidence
        super().__init__(reason)


class AgentExecutionError(Exception):
    pass


class BaseAgent(abc.ABC):
    """Abstract base agent class"""

    def __init__(self):
        self.llm = LLMClient()

    async def _log_step(
        self,
        context: AgentContext,
        step_type: str,
        description: str,
        observation: Optional[str] = None,
        confidence: Optional[float] = None
    ) -> None:
        """Log a single execution step into the context"""
        context.current_step += 1
        context.step_trace.append({
            "step": context.current_step,
            "type": step_type,
            "description": description,
            "observation": observation,
            "confidence": confidence
        })
        if confidence is not None:
            context.confidence_scores.append(confidence)

    async def _fallback(self, reason: str, suggestion: Optional[str] = None, confidence: float = 0.0):
        """Trigger a fallback with explanation and confidence"""
        raise FallbackTrigger(reason=reason, suggestion=suggestion, confidence=confidence)

    @abc.abstractmethod
    async def execute_logic(self, context: AgentContext) -> str:
        """Core logic to be implemented by all agents"""
        pass
