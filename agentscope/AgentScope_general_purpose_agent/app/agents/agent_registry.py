"""
💡 What This Is:
A centralized registry that dynamically maps agent names like "refund" or "fraud" to Python classes.

🧠 Why We Need This:
CTO: “How do I add a new agent without rewriting 300 lines?”
This solves extensibility — just register a class, no core code change
Enables LLM agents, rule-based agents, even hybrid agents

⏰ When It’s Used:
During agent execution: agent = AgentRegistry().get_agent("refund")

🛠️ How It Works:
Registry stores agent classes
Can be dynamically extended at runtime or via subclassing
"""

from typing import Dict, Type
from app.agents.base_agent import BaseAgent

class AgentRegistry:
    """
    Global registry for agent classes.
    Supports dynamic addition of new agents.
    """

    def __init__(self):
        self._registry: Dict[str, Type[BaseAgent]] = {}

    def register(self, agent_name: str, agent_cls: Type[BaseAgent]):
        if not issubclass(agent_cls, BaseAgent):
            raise ValueError("All agents must subclass BaseAgent")
        self._registry[agent_name.lower()] = agent_cls

    def get_agent(self, agent_name: str) -> BaseAgent:
        cls = self._registry.get(agent_name.lower())
        if not cls:
            raise ValueError(f"Agent type '{agent_name}' is not registered.")
        return cls()

    def list_agents(self):
        return list(self._registry.keys())
