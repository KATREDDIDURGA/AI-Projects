# app/models/agent_step.py
from sqlalchemy import Column, String, Text, DateTime, ForeignKey, Integer, Float
from app.db.base import Base
import datetime
import uuid


class AgentStep(Base):
    __tablename__ = "agent_steps"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    run_id = Column(String, ForeignKey("agent_runs.run_id"), nullable=False)
    step_number = Column(Integer, nullable=False)
    step_type = Column(String, nullable=False)
    description = Column(Text)
    observation = Column(Text, nullable=True)
    confidence = Column(Float, nullable=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "run_id": self.run_id,
            "step_number": self.step_number,
            "step_type": self.step_type,
            "description": self.description,
            "observation": self.observation,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }