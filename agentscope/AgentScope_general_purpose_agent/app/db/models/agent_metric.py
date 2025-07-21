from sqlalchemy import Column, Integer, String, Float, ForeignKey
from app.db.base import Base



class AgentMetric(Base):
    __tablename__ = "agent_metrics"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String, ForeignKey("agent_runs.run_id"))
    metric_name = Column(String)
    value = Column(Float)

    def to_dict(self):
        return {
            "id": self.id,
            "run_id": self.run_id,
            "metric_name": self.metric_name,
            "value": self.value
        }
