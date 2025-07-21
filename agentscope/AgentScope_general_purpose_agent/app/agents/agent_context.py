'''nts stateless — context carries the state

⏰ When It’s Used:
Created once for each agent execution and passed through the full lifecycle.

🛠️ How It Works:
Holds query, metadata, decision graph, step trace, and score

Automatically builds step-wise graph for visualization

'''

from typing import Optional, List, Dict, Any
from app.utils.decision_graph import DecisionGraph


class AgentContext:
    def __init__(self, run_id: str, user_query: str, metadata: Optional[Dict[str, Any]] = None):
        self.run_id = run_id
        self.user_query = user_query
        self.metadata = metadata or {}

        # Execution tracking
        self.current_step: int = 0
        self.step_trace: List[Dict[str, Any]] = []
        self.confidence_scores: List[float] = []

        # Graph trace builder
        self.graph = DecisionGraph(run_id)

    @property
    def overall_confidence(self) -> float:
        """Average confidence over all steps (if applicable)"""
        if not self.confidence_scores:
            return 0.0
        return round(sum(self.confidence_scores) / len(self.confidence_scores), 2)
