"""
AgentScope Base Agent Framework

Abstract base class for all AI agents with built-in monitoring,
state management, and standardized execution patterns.
"""

import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from enum import Enum

from app.core.config import settings
from app.core.database import db_manager, AgentRun, AgentStep
from app.models.schemas import AgentStatus, StepType, ConfidenceLevel

# Configure logging
logger = logging.getLogger(__name__)


class AgentExecutionError(Exception):
    """Custom exception for agent execution errors"""
    def __init__(self, message: str, step_number: Optional[int] = None, details: Optional[Dict] = None):
        self.message = message
        self.step_number = step_number
        self.details = details or {}
        super().__init__(self.message)


class FallbackTrigger(Exception):
    """Exception to trigger fallback behavior"""
    def __init__(self, reason: str, confidence: float, suggestion: Optional[str] = None, critical: bool = False):
        self.reason = reason
        self.confidence = confidence
        self.suggestion = suggestion
        self.critical = critical  # New: indicates if this should block execution
        super().__init__(f"Fallback triggered: {reason}")


class BusinessRuleViolation(FallbackTrigger):
    """Exception for business rule violations that should block execution"""
    def __init__(self, reason: str, rule_violated: str, evidence: Dict[str, Any]):
        self.rule_violated = rule_violated
        self.evidence = evidence
        super().__init__(reason, confidence=0.0, suggestion="Human review required", critical=True)


class AgentContext:
    """Context object passed through agent execution"""
    
    def __init__(self, run_id: str, query: str, metadata: Optional[Dict] = None):
        self.run_id = run_id
        self.query = query
        self.metadata = metadata or {}
        
        # Execution state
        self.current_step = 0
        self.start_time = time.time()
        self.step_start_time = time.time()
        
        # Confidence tracking
        self.confidence_scores: List[float] = []
        self.overall_confidence: Optional[float] = None
        
        # Result tracking
        self.observations: List[Dict] = []
        self.final_result: Optional[str] = None
        self.fallback_triggered = False
        self.fallback_reason: Optional[str] = None
        
        # Performance metrics
        self.step_durations: List[int] = []
        self.ai_api_calls = 0
        self.database_queries = 0
    
    @property
    def execution_time_ms(self) -> int:
        """Get current execution time in milliseconds"""
        return int((time.time() - self.start_time) * 1000)
    
    @property
    def average_confidence(self) -> float:
        """Calculate average confidence across all steps"""
        if not self.confidence_scores:
            return 0.0
        return sum(self.confidence_scores) / len(self.confidence_scores)
    
    def add_confidence_score(self, score: float) -> None:
        """Add a confidence score and update overall confidence"""
        self.confidence_scores.append(score)
        self.overall_confidence = self.average_confidence


class BaseAgent(ABC):
    """
    Abstract base class for all AgentScope agents.
    
    Provides standardized execution framework with:
    - State management and persistence
    - Confidence monitoring and fallback detection
    - Performance tracking and metrics
    - Error handling and recovery
    """
    
    def __init__(self, agent_type: str):
        self.agent_type = agent_type
        self.logger = logging.getLogger(f"{__name__}.{agent_type}")
        
        # Configuration from settings
        self.min_confidence_threshold = settings.agents.MIN_CONFIDENCE_THRESHOLD
        self.high_confidence_threshold = settings.agents.HIGH_CONFIDENCE_THRESHOLD
        self.fallback_confidence_threshold = settings.agents.FALLBACK_CONFIDENCE_THRESHOLD
        self.max_steps = settings.agents.MAX_AGENT_STEPS
        self.step_timeout = settings.agents.AGENT_STEP_TIMEOUT
        self.total_timeout = settings.agents.AGENT_TOTAL_TIMEOUT
    
    @abstractmethod
    async def execute_logic(self, context: AgentContext) -> str:
        """
        Main agent logic implementation.
        
        Args:
            context: Agent execution context with query and state
            
        Returns:
            Final decision/result string
            
        Raises:
            FallbackTrigger: When agent should fallback
            AgentExecutionError: On execution errors
        """
        pass
    
    @abstractmethod
    def get_agent_description(self) -> str:
        """Return a description of what this agent does"""
        pass
    
    async def execute(self, query: str, metadata: Optional[Dict] = None) -> str:
        """
        Execute the agent with full monitoring and state management.
        
        Args:
            query: User query to process
            metadata: Optional metadata for execution context
            
        Returns:
            Run ID for tracking execution
        """
        run_id = str(uuid.uuid4())
        context = AgentContext(run_id, query, metadata)
        
        try:
            # Initialize database record
            await db_manager.create_agent_run(
                run_id=run_id,
                query=query,
                agent_type=self.agent_type
            )
            
            self.logger.info(f"Starting agent execution: {run_id}")
            
            # Execute with timeout
            result = await asyncio.wait_for(
                self._execute_with_monitoring(context),
                timeout=self.total_timeout
            )
            
            # Complete the run
            await db_manager.complete_agent_run(
                run_id=run_id,
                final_decision=result,
                confidence_score=context.overall_confidence or 0.0,
                total_steps=context.current_step,
                execution_time_ms=context.execution_time_ms,
                status=AgentStatus.COMPLETED
            )
            
            self.logger.info(f"Agent execution completed: {run_id}")
            return run_id
            
        except asyncio.TimeoutError:
            await self._handle_timeout(context)
            raise AgentExecutionError(f"Agent execution timed out after {self.total_timeout}s")
            
        except FallbackTrigger as e:
            # Handle different types of fallbacks
            if hasattr(e, 'critical') and e.critical:
                # Critical fallbacks should prevent execution
                await self._handle_critical_fallback(context, e)
                raise BusinessRuleViolation(
                    reason=f"Execution blocked: {e.reason}",
                    rule_violated="CRITICAL_FALLBACK",
                    evidence={"original_reason": e.reason, "confidence": e.confidence}
                )
            else:
                # Non-critical fallbacks can continue with warnings
                await self._handle_fallback(context, e)
                raise
            
        except BusinessRuleViolation as e:
            await self._handle_business_rule_violation(context, e)
            raise
            
        except Exception as e:
            await self._handle_error(context, e)
            raise
    
    async def _execute_with_monitoring(self, context: AgentContext) -> str:
        """Execute agent logic with step monitoring and confidence tracking"""
        try:
            # Log initial thought
            await self._log_step(
                context,
                StepType.THOUGHT,
                "Starting agent execution",
                {"query": context.query, "agent_type": self.agent_type}
            )
            
            # Execute main agent logic
            result = await self.execute_logic(context)
            
            # Log final decision
            await self._log_step(
                context,
                StepType.DECISION,
                "Agent execution completed",
                {"final_decision": result, "confidence": context.overall_confidence}
            )
            
            context.final_result = result
            return result
            
        except FallbackTrigger:
            # Re-raise fallback triggers
            raise
            
        except Exception as e:
            # Convert other exceptions to agent errors
            raise AgentExecutionError(f"Execution failed: {str(e)}", context.current_step)
    
    async def _log_step(
        self,
        context: AgentContext,
        step_type: StepType,
        description: str,
        observation: Optional[Dict] = None,
        confidence: Optional[float] = None
    ) -> None:
        """Log a step in the agent's reasoning process"""
        context.current_step += 1
        
        # Calculate step duration
        current_time = time.time()
        step_duration_ms = int((current_time - context.step_start_time) * 1000)
        context.step_durations.append(step_duration_ms)
        context.step_start_time = current_time
        
        # Add confidence score if provided
        if confidence is not None:
            context.add_confidence_score(confidence)
            
            # Check for fallback conditions
            await self._check_fallback_conditions(context, confidence)
        
        # Store observation
        if observation:
            context.observations.append(observation)
        
        # Log to database
        await db_manager.add_agent_step(
            run_id=context.run_id,
            step_number=context.current_step,
            step_type=step_type.value,
            description=description,
            observation=json.dumps(observation) if observation else None,
            confidence=confidence
        )
        
        # Check step limits
        if context.current_step >= self.max_steps:
            raise AgentExecutionError(f"Maximum steps exceeded: {self.max_steps}")
        
        self.logger.debug(f"Step {context.current_step}: {step_type} - {description}")
    
    async def _check_fallback_conditions(self, context: AgentContext, confidence: float) -> None:
        """Check if fallback should be triggered based on confidence"""
        if confidence < self.fallback_confidence_threshold:
            raise FallbackTrigger(
                reason=f"Confidence {confidence:.2f} below threshold {self.fallback_confidence_threshold}",
                confidence=confidence,
                suggestion="Consider manual review or alternative approach",
                critical=True  # Low confidence should block execution
            )
    
    async def validate_business_rules(self, context: AgentContext, decision_data: Dict[str, Any]) -> None:
        """Validate business rules before finalizing decisions"""
        # This method should be overridden by specific agents
        pass
    
    async def check_decision_consistency(self, context: AgentContext, final_decision: str, confidence: float) -> None:
        """Check if the final decision is consistent with the evidence"""
        # Check for low confidence on critical decisions
        if "approved" in final_decision.lower() and confidence < 0.7:
            raise BusinessRuleViolation(
                reason=f"High-risk approval with low confidence ({confidence:.2f})",
                rule_violated="CONFIDENCE_THRESHOLD_APPROVAL",
                evidence={"decision": final_decision, "confidence": confidence}
            )
        
        # Check for missing confidence scores
        if confidence is None or confidence == 0:
            raise BusinessRuleViolation(
                reason="Decision made without confidence calculation",
                rule_violated="MISSING_CONFIDENCE",
                evidence={"decision": final_decision, "confidence": confidence}
            )
    
    async def _handle_fallback(self, context: AgentContext, fallback: FallbackTrigger) -> None:
        """Handle fallback trigger"""
        context.fallback_triggered = True
        context.fallback_reason = fallback.reason
        
        # Log fallback step
        await self._log_step(
            context,
            StepType.FALLBACK,
            f"Fallback triggered: {fallback.reason}",
            {
                "reason": fallback.reason,
                "confidence": fallback.confidence,
                "suggestion": fallback.suggestion
            }
        )
        
        # Mark run as completed with fallback
        await db_manager.complete_agent_run(
            run_id=context.run_id,
            final_decision=f"Fallback: {fallback.reason}",
            confidence_score=fallback.confidence,
            total_steps=context.current_step,
            execution_time_ms=context.execution_time_ms,
            status=AgentStatus.COMPLETED
        )
        
    async def _handle_critical_fallback(self, context: AgentContext, fallback: FallbackTrigger) -> None:
        """Handle critical fallback that blocks execution"""
        context.fallback_triggered = True
        context.fallback_reason = fallback.reason
        
        # Log critical fallback
        await self._log_step(
            context,
            StepType.FALLBACK,
            f"🚨 CRITICAL FALLBACK - Execution Blocked: {fallback.reason}",
            {
                "reason": fallback.reason,
                "confidence": fallback.confidence,
                "suggestion": fallback.suggestion,
                "severity": "CRITICAL",
                "action_required": "HUMAN_REVIEW"
            }
        )
        
        # Mark run as requiring human review
        await db_manager.complete_agent_run(
            run_id=context.run_id,
            final_decision=f"🚨 BLOCKED - Human Review Required: {fallback.reason}",
            confidence_score=fallback.confidence,
            total_steps=context.current_step,
            execution_time_ms=context.execution_time_ms,
            status="blocked_for_review"  # New status
        )
        
        self.logger.critical(f"CRITICAL FALLBACK - Execution blocked for {context.run_id}: {fallback.reason}")
    
    async def _handle_business_rule_violation(self, context: AgentContext, violation: BusinessRuleViolation) -> None:
        """Handle business rule violations"""
        # Log the violation
        await self._log_step(
            context,
            StepType.FALLBACK,
            f"⛔ BUSINESS RULE VIOLATION: {violation.rule_violated}",
            {
                "rule_violated": violation.rule_violated,
                "reason": violation.reason,
                "evidence": violation.evidence,
                "severity": "CRITICAL",
                "action_required": "IMMEDIATE_HUMAN_REVIEW"
            }
        )
        
        # Mark as blocked
        await db_manager.complete_agent_run(
            run_id=context.run_id,
            final_decision=f"⛔ RULE VIOLATION - {violation.rule_violated}: {violation.reason}",
            confidence_score=0.0,
            total_steps=context.current_step,
            execution_time_ms=context.execution_time_ms,
            status="rule_violation"
        )
        
        self.logger.error(f"BUSINESS RULE VIOLATION for {context.run_id}: {violation.rule_violated} - {violation.reason}")
    
    async def _handle_timeout(self, context: AgentContext) -> None:
        """Handle execution timeout"""
        await db_manager.mark_agent_run_failed(
            run_id=context.run_id,
            error_message=f"Execution timed out after {self.total_timeout}s",
            error_step=context.current_step
        )
        
        self.logger.error(f"Agent execution timed out: {context.run_id}")
    
    async def _handle_error(self, context: AgentContext, error: Exception) -> None:
        """Handle execution error"""
        error_step = context.current_step if hasattr(error, 'step_number') and error.step_number else context.current_step
        
        await db_manager.mark_agent_run_failed(
            run_id=context.run_id,
            error_message=str(error),
            error_step=error_step
        )
        
        self.logger.error(f"Agent execution failed: {context.run_id} - {error}")
    
    # ================================
    # HELPER METHODS FOR AGENT LOGIC
    # ================================
    
    async def thought(self, context: AgentContext, description: str, details: Optional[Dict] = None) -> None:
        """Log a reasoning thought"""
        await self._log_step(context, StepType.THOUGHT, description, details)
    
    async def observation(
        self,
        context: AgentContext,
        description: str,
        data: Optional[Dict] = None,
        confidence: Optional[float] = None
    ) -> None:
        """Log an observation with optional confidence"""
        await self._log_step(context, StepType.OBSERVATION, description, data, confidence)
    
    async def action(self, context: AgentContext, description: str, result: Optional[Dict] = None) -> None:
        """Log an action taken"""
        await self._log_step(context, StepType.ACTION, description, result)
    
    async def calculate_confidence(self, *factors: float) -> float:
        """Calculate confidence score from multiple factors"""
        if not factors:
            return 0.0
        
        # Weighted average with some factors being more important
        weights = [1.0] * len(factors)  # Equal weights by default
        
        weighted_sum = sum(f * w for f, w in zip(factors, weights))
        total_weight = sum(weights)
        
        confidence = weighted_sum / total_weight
        return max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
    
    async def check_confidence_threshold(self, confidence: float, threshold: Optional[float] = None) -> bool:
        """Check if confidence meets threshold"""
        threshold = threshold or self.min_confidence_threshold
        return confidence >= threshold
    
    async def track_api_call(self, context: AgentContext, service: str, duration_ms: int) -> None:
        """Track external API calls for performance monitoring"""
        context.ai_api_calls += 1
        
        await db_manager.add_metric(
            run_id=context.run_id,
            metric_name=f"api_call_{service}",
            metric_value=duration_ms,
            metric_unit="milliseconds",
            step_number=context.current_step
        )
    
    async def track_database_query(self, context: AgentContext, query_type: str, duration_ms: int) -> None:
        """Track database queries for performance monitoring"""
        context.database_queries += 1
        
        await db_manager.add_metric(
            run_id=context.run_id,
            metric_name=f"db_query_{query_type}",
            metric_value=duration_ms,
            metric_unit="milliseconds",
            step_number=context.current_step
        )
    
    @asynccontextmanager
    async def timed_operation(self, context: AgentContext, operation_name: str):
        """Context manager for timing operations"""
        start_time = time.time()
        try:
            yield
        finally:
            duration_ms = int((time.time() - start_time) * 1000)
            await db_manager.add_metric(
                run_id=context.run_id,
                metric_name=f"operation_{operation_name}",
                metric_value=duration_ms,
                metric_unit="milliseconds",
                step_number=context.current_step
            )
    
    # ================================
    # AGENT INTROSPECTION
    # ================================
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get agent configuration for debugging"""
        return {
            "agent_type": self.agent_type,
            "min_confidence_threshold": self.min_confidence_threshold,
            "high_confidence_threshold": self.high_confidence_threshold,
            "fallback_confidence_threshold": self.fallback_confidence_threshold,
            "max_steps": self.max_steps,
            "step_timeout": self.step_timeout,
            "total_timeout": self.total_timeout,
        }
    
    def get_capabilities(self) -> List[str]:
        """Get list of agent capabilities"""
        return [
            "confidence_monitoring",
            "fallback_detection",
            "step_tracking",
            "performance_metrics",
            "error_recovery",
            "timeout_handling"
        ]