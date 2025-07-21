# app/models/agent_run.py
from sqlalchemy import Column, String, DateTime, Text
from app.db.base import Base
import datetime


class AgentRun(Base):
    __tablename__ = "agent_runs"

    run_id = Column(String, primary_key=True, index=True)
    agent_type = Column(String)  # This field was missing in your original
    query = Column(Text)
    status = Column(String)
    started_at = Column(DateTime, default=datetime.datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    final_decision = Column(Text, nullable=True)

    def to_dict(self):
        return {
            "run_id": self.run_id,
            "agent_type": self.agent_type,
            "query": self.query,
            "status": self.status,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "final_decision": self.final_decision
        }