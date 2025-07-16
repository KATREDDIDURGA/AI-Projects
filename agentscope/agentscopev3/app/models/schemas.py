"""
AgentScope API Data Models

Pydantic schemas for request/response validation, serialization,
and automatic API documentation generation.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum

from pydantic import BaseModel, Field, validator, ConfigDict
from pydantic.types import UUID4


# ================================
# ENUMS FOR CONSISTENT VALUES
# ================================

class AgentStatus(str, Enum):
    """Agent execution status"""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    BLOCKED_FOR_REVIEW = "blocked_for_review"  # New status
    RULE_VIOLATION = "rule_violation"  # New status


class StepType(str, Enum):
    """Types of agent reasoning steps"""
    THOUGHT = "Thought"
    OBSERVATION = "Observation"
    DECISION = "Decision"
    FALLBACK = "Fallback"
    ACTION = "Action"


class AgentType(str, Enum):
    """Supported agent types"""
    REFUND_AGENT = "refund_agent"
    FRAUD_AGENT = "fraud_agent"
    GENERIC_AGENT = "generic_agent"


class ConfidenceLevel(str, Enum):
    """Confidence level categories"""
    LOW = "low"          # 0.0 - 0.3
    MEDIUM = "medium"    # 0.3 - 0.7
    HIGH = "high"        # 0.7 - 1.0


# ================================
# BASE SCHEMAS
# ================================

class BaseSchema(BaseModel):
    """Base schema with common configuration"""
    model_config = ConfigDict(
        from_attributes=True,
        use_enum_values=True,
        validate_assignment=True,
        arbitrary_types_allowed=True
    )


class TimestampMixin(BaseModel):
    """Mixin for timestamp fields"""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None


# ================================
# REQUEST SCHEMAS
# ================================

class AgentRunRequest(BaseSchema):
    """Request to start a new agent execution"""
    query: str = Field(
        ..., 
        min_length=1, 
        max_length=1000,
        description="User query for the agent to process",
        example="I want a refund for my gaming mouse that broke after 30 days"
    )
    agent_type: AgentType = Field(
        default=AgentType.REFUND_AGENT,
        description="Type of agent to execute"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional context or configuration"
    )
    
    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty or whitespace only')
        return v.strip()


class StepUpdateRequest(BaseSchema):
    """Request to update a specific step"""
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    observation: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


# ================================
# RESPONSE SCHEMAS
# ================================

class AgentStepResponse(BaseSchema):
    """Individual agent reasoning step"""
    step_number: int = Field(..., ge=1, description="Sequential step number")
    step_type: StepType = Field(..., description="Type of reasoning step")
    description: str = Field(..., description="Human-readable step description")
    observation: Optional[Dict[str, Any]] = Field(None, description="Step observation data")
    
    # Timing information
    started_at: datetime = Field(..., description="When step started")
    completed_at: Optional[datetime] = Field(None, description="When step completed")
    duration_ms: Optional[int] = Field(None, ge=0, description="Step duration in milliseconds")
    
    # Quality metrics
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Step confidence score")
    success: Optional[bool] = Field(None, description="Whether step succeeded")
    
    @property
    def confidence_level(self) -> Optional[ConfidenceLevel]:
        """Get confidence level category"""
        if self.confidence is None:
            return None
        elif self.confidence < 0.3:
            return ConfidenceLevel.LOW
        elif self.confidence < 0.7:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.HIGH


class AgentMetricResponse(BaseSchema):
    """Performance metric for an agent run"""
    metric_name: str = Field(..., description="Metric identifier")
    metric_value: float = Field(..., description="Metric value")
    metric_unit: Optional[str] = Field(None, description="Unit of measurement")
    step_number: Optional[int] = Field(None, description="Associated step number")
    timestamp: datetime = Field(..., description="When metric was recorded")
    extra_data: Optional[Dict[str, Any]] = Field(None, description="Additional metric context")


class AgentRunResponse(BaseSchema):
    """Complete agent execution run"""
    # Basic identification
    id: str = Field(..., description="Unique run identifier")
    query: str = Field(..., description="Original user query")
    agent_type: AgentType = Field(..., description="Agent type executed")
    
    # Execution status
    status: AgentStatus = Field(..., description="Current execution status")
    started_at: datetime = Field(..., description="Execution start time")
    completed_at: Optional[datetime] = Field(None, description="Execution completion time")
    
    # Results
    final_decision: Optional[str] = Field(None, description="Final agent decision")
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Overall confidence")
    
    # Error information
    error_message: Optional[str] = Field(None, description="Error message if failed")
    error_step: Optional[int] = Field(None, description="Step where error occurred")
    
    # Performance metrics
    total_steps: int = Field(default=0, ge=0, description="Total number of steps")
    execution_time_ms: Optional[int] = Field(None, ge=0, description="Total execution time")
    
    # Related data
    steps: List[AgentStepResponse] = Field(default=[], description="All execution steps")
    metrics: List[AgentMetricResponse] = Field(default=[], description="Performance metrics")
    
    @property
    def confidence_level(self) -> Optional[ConfidenceLevel]:
        """Get overall confidence level"""
        if self.confidence_score is None:
            return None
        elif self.confidence_score < 0.3:
            return ConfidenceLevel.LOW
        elif self.confidence_score < 0.7:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.HIGH
    
    @property
    def is_completed(self) -> bool:
        """Check if run is completed (success or failure)"""
        return self.status in [AgentStatus.COMPLETED, AgentStatus.FAILED, AgentStatus.TIMEOUT]
    
    @property
    def execution_time_seconds(self) -> Optional[float]:
        """Get execution time in seconds"""
        return self.execution_time_ms / 1000.0 if self.execution_time_ms else None


class AgentRunSummaryResponse(BaseSchema):
    """Lightweight agent run summary for lists"""
    id: str
    query: str
    agent_type: AgentType
    status: AgentStatus
    started_at: datetime
    completed_at: Optional[datetime]
    final_decision: Optional[str]
    confidence_score: Optional[float]
    total_steps: int
    execution_time_ms: Optional[int]
    
    @property
    def confidence_level(self) -> Optional[ConfidenceLevel]:
        if self.confidence_score is None:
            return None
        elif self.confidence_score < 0.3:
            return ConfidenceLevel.LOW
        elif self.confidence_score < 0.7:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.HIGH


# ================================
# REAL-TIME UPDATE SCHEMAS
# ================================

class StepStreamUpdate(BaseSchema):
    """Real-time step update for WebSocket streaming"""
    run_id: str
    step: AgentStepResponse
    total_steps: int
    is_final: bool = False


class RunStatusUpdate(BaseSchema):
    """Real-time run status update"""
    run_id: str
    status: AgentStatus
    progress_percentage: Optional[float] = Field(None, ge=0.0, le=100.0)
    current_step: Optional[int] = None
    estimated_remaining_ms: Optional[int] = None


class ConfidenceUpdate(BaseSchema):
    """Real-time confidence score update"""
    run_id: str
    step_number: int
    confidence: float = Field(..., ge=0.0, le=1.0)
    confidence_level: ConfidenceLevel
    threshold_breach: Optional[str] = Field(None, description="Which threshold was breached")


# ================================
# ANALYTICS & REPORTING SCHEMAS
# ================================

class ConfidenceStatistics(BaseSchema):
    """Confidence score statistics"""
    avg_confidence: float = Field(..., ge=0.0, le=1.0)
    min_confidence: float = Field(..., ge=0.0, le=1.0)
    max_confidence: float = Field(..., ge=0.0, le=1.0)
    total_runs: int = Field(..., ge=0)
    confidence_distribution: Dict[ConfidenceLevel, int] = Field(default={})


class PerformanceMetrics(BaseSchema):
    """Agent performance metrics"""
    avg_execution_time_ms: float = Field(..., ge=0)
    avg_steps_per_run: float = Field(..., ge=0)
    success_rate: float = Field(..., ge=0.0, le=1.0)
    fallback_rate: float = Field(..., ge=0.0, le=1.0)
    total_runs: int = Field(..., ge=0)


class SystemHealth(BaseSchema):
    """Overall system health metrics"""
    active_runs: int = Field(..., ge=0)
    completed_runs_today: int = Field(..., ge=0)
    avg_confidence_today: Optional[float] = Field(None, ge=0.0, le=1.0)
    error_rate_today: float = Field(..., ge=0.0, le=1.0)
    uptime_seconds: int = Field(..., ge=0)


# ================================
# ERROR RESPONSE SCHEMAS
# ================================

class ErrorDetail(BaseSchema):
    """Detailed error information"""
    code: str = Field(..., description="Error code identifier")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error context")


class ValidationErrorResponse(BaseSchema):
    """Validation error response"""
    error_type: str = "validation_error"
    message: str = "Request validation failed"
    errors: List[ErrorDetail] = Field(..., description="List of validation errors")


class AgentErrorResponse(BaseSchema):
    """Agent execution error response"""
    error_type: str = "agent_error"
    message: str = Field(..., description="Error description")
    run_id: Optional[str] = Field(None, description="Associated run ID")
    step_number: Optional[int] = Field(None, description="Step where error occurred")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")


# ================================
# API RESPONSE WRAPPERS
# ================================

class APIResponse(BaseSchema):
    """Standard API response wrapper"""
    success: bool = Field(..., description="Whether request succeeded")
    message: str = Field(..., description="Response message")
    data: Optional[Any] = Field(None, description="Response data")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PaginatedResponse(BaseSchema):
    """Paginated response wrapper"""
    items: List[Any] = Field(..., description="Response items")
    total: int = Field(..., ge=0, description="Total items available")
    page: int = Field(..., ge=1, description="Current page number")
    per_page: int = Field(..., ge=1, description="Items per page")
    pages: int = Field(..., ge=1, description="Total pages")
    has_next: bool = Field(..., description="Whether next page exists")
    has_prev: bool = Field(..., description="Whether previous page exists")


# ================================
# UTILITY FUNCTIONS
# ================================

def create_success_response(data: Any = None, message: str = "Success") -> APIResponse:
    """Create a successful API response"""
    return APIResponse(
        success=True,
        message=message,
        data=data
    )


def create_error_response(message: str, data: Any = None) -> APIResponse:
    """Create an error API response"""
    return APIResponse(
        success=False,
        message=message,
        data=data
    )


def agent_run_to_response(agent_run) -> AgentRunResponse:
    """Convert database AgentRun to response schema"""
    from app.core.database import AgentRun, AgentStep, AgentMetric
    
    # Convert steps
    steps = []
    if hasattr(agent_run, 'steps') and agent_run.steps:
        for step in sorted(agent_run.steps, key=lambda x: x.step_number):
            steps.append(AgentStepResponse(
                step_number=step.step_number,
                step_type=step.step_type,
                description=step.description,
                observation=step.observation,
                started_at=step.started_at,
                completed_at=step.completed_at,
                duration_ms=step.duration_ms,
                confidence=step.confidence,
                success=step.success
            ))
    
    # Convert metrics
    metrics = []
    if hasattr(agent_run, 'metrics') and agent_run.metrics:
        for metric in agent_run.metrics:
            metrics.append(AgentMetricResponse(
                metric_name=metric.metric_name,
                metric_value=metric.metric_value,
                metric_unit=metric.metric_unit,
                step_number=metric.step_number,
                timestamp=metric.timestamp,
                extra_data=metric.extra_data
            ))
    
    return AgentRunResponse(
        id=agent_run.id,
        query=agent_run.query,
        agent_type=agent_run.agent_type,
        status=agent_run.status,
        started_at=agent_run.started_at,
        completed_at=agent_run.completed_at,
        final_decision=agent_run.final_decision,
        confidence_score=agent_run.confidence_score,
        error_message=agent_run.error_message,
        error_step=agent_run.error_step,
        total_steps=agent_run.total_steps,
        execution_time_ms=agent_run.execution_time_ms,
        steps=steps,
        metrics=metrics
    )