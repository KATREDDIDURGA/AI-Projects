'''💡 What This Is:
A class that converts each agent step into a graph structure — useful for later visualizations and audits.

🧠 Why We Need This:
The CTO asked: "Where is the decision graph you promised?"

This will let us trace logic like:

pgsql
Copy
Edit
User Query ➝ Intent ➝ Policy Check ➝ Rule Violation ➝ Final Decision
⏰ When It’s Used:
Every time a step is logged (log_step() in BaseAgent), it adds a node/edge here.

🛠️ How It Works:
Stores nodes with type, timestamp, confidence

Stores edges with step relationships

'''

from typing import Dict, List, Optional
from datetime import datetime


class DecisionGraph:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.nodes: List[Dict] = []
        self.edges: List[Dict] = []

    def add_node(self, step_number: int, step_type: str, description: str,
                 observation: Optional[str] = None, confidence: Optional[float] = None) -> None:
        node = {
            "id": f"{self.run_id}_step_{step_number}",
            "step": step_number,
            "type": step_type,
            "description": description,
            "observation": observation,
            "confidence": confidence,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.nodes.append(node)

        # Link to previous step if exists
        if step_number > 1:
            self.edges.append({
                "from": f"{self.run_id}_step_{step_number - 1}",
                "to": f"{self.run_id}_step_{step_number}"
            })

    def export(self) -> Dict[str, List[Dict]]:
        """Export graph data as a dict"""
        return {
            "nodes": self.nodes,
            "edges": self.edges
        }
