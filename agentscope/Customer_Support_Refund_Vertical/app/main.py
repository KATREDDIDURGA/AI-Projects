"""
AgentScope Main Application

Production-ready FastAPI application with comprehensive agent execution,
real-time monitoring, and debugging capabilities.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, List, Optional, Any

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.websockets import WebSocket
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from starlette.status import HTTP_404_NOT_FOUND, HTTP_500_INTERNAL_SERVER_ERROR

from app.core.config import settings
from app.core.database import db_manager, get_database
from app.services.execution_service import execution_service
from app.api.websocket import websocket_router, connection_manager, get_websocket_health
from app.agents.refund_agent import RefundAgent
from app.models.schemas import (
    AgentRunRequest, AgentRunResponse, AgentRunSummaryResponse,
    APIResponse, PaginatedResponse, SystemHealth, PerformanceMetrics,
    AgentStatus, create_success_response, create_error_response,
    agent_run_to_response
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.monitoring.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Application lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown"""
    # Startup
    logger.info("Starting AgentScope application...")
    
    try:
        # Initialize database
        await db_manager.initialize()
        logger.info("Database initialized")
        
        # Initialize execution service
        await execution_service.initialize()
        logger.info("Execution service initialized")
        
        # Set startup time for health checks
        app.state.startup_time = datetime.utcnow()
        app.state.healthy = True
        
        logger.info("AgentScope application started successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise
    
    finally:
        # Shutdown
        logger.info("Shutting down AgentScope application...")
        
        try:
            # Shutdown execution service
            await execution_service.shutdown()
            logger.info("Execution service shutdown")
            
            # Close database connections
            await db_manager.close()
            logger.info("Database connections closed")
            
            app.state.healthy = False
            logger.info("AgentScope application shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


# Create FastAPI application
app = FastAPI(
    title=settings.app.APP_NAME,
    description=settings.app.APP_DESCRIPTION,
    version=settings.app.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs" if settings.app.DEBUG else None,
    redoc_url="/redoc" if settings.app.DEBUG else None,
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.security.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Include WebSocket router
app.include_router(websocket_router)


# ================================
# AGENT EXECUTION ENDPOINTS
# ================================

@app.post("/api/v1/agents/execute", response_model=APIResponse)
async def execute_agent(
    request: AgentRunRequest,
    background_tasks: BackgroundTasks
) -> APIResponse:
    """
    Start a new agent execution
    
    Returns immediately with a run_id for tracking progress via WebSocket
    """
    try:
        # Validate agent type
        if request.agent_type not in execution_service.agent_registry.list_agent_types():
            raise HTTPException(
                status_code=400,
                detail=f"Unknown agent type: {request.agent_type}"
            )
        
        # Start execution
        run_id = await execution_service.start_agent_execution(
            agent_type=request.agent_type,
            query=request.query,
            metadata=request.metadata
        )
        
        # Log execution start
        logger.info(f"Started agent execution: {run_id} ({request.agent_type})")
        
        return create_success_response(
            data={
                "run_id": run_id,
                "agent_type": request.agent_type,
                "status": "started",
                "websocket_url": f"/ws/runs/{run_id}",
                "polling_url": f"/api/v1/runs/{run_id}"
            },
            message="Agent execution started successfully"
        )
        
    except Exception as e:
        logger.error(f"Failed to start agent execution: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/runs/{run_id}", response_model=AgentRunResponse)
async def get_agent_run(run_id: str) -> AgentRunResponse:
    """
    Get complete agent run details including all steps and metrics
    """
    try:
        # Get from database
        agent_run = await db_manager.get_agent_run(run_id)
        if not agent_run:
            raise HTTPException(status_code=404, detail="Agent run not found")
        
        # Convert to response format
        return agent_run_to_response(agent_run)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get agent run {run_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/v1/runs/{run_id}/status")
async def get_execution_status(run_id: str) -> Dict[str, Any]:
    """
    Get current execution status (lightweight endpoint for polling fallback)
    """
    try:
        status = await execution_service.get_execution_status(run_id)
        if not status:
            raise HTTPException(status_code=404, detail="Run not found")
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get execution status for {run_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.delete("/api/v1/runs/{run_id}/cancel")
async def cancel_execution(run_id: str) -> APIResponse:
    """
    Cancel an active agent execution
    """
    try:
        success = await execution_service.cancel_execution(run_id)
        
        if success:
            return create_success_response(
                data={"run_id": run_id, "cancelled": True},
                message="Execution cancelled successfully"
            )
        else:
            return create_error_response(
                message="Execution not found or already completed",
                data={"run_id": run_id, "cancelled": False}
            )
            
    except Exception as e:
        logger.error(f"Failed to cancel execution {run_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/v1/runs", response_model=PaginatedResponse)
async def list_agent_runs(
    page: int = 1,
    per_page: int = 20,
    status: Optional[AgentStatus] = None
) -> PaginatedResponse:
    """
    List recent agent runs with pagination and filtering
    """
    try:
        # Validate pagination
        if page < 1 or per_page < 1 or per_page > 100:
            raise HTTPException(status_code=400, detail="Invalid pagination parameters")
        
        # Get runs from database
        recent_runs = await db_manager.get_recent_runs(limit=per_page * 5)  # Get more for filtering
        
        # Filter by status if specified
        if status:
            recent_runs = [run for run in recent_runs if run.status == status.value]
        
        # Paginate
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        page_runs = recent_runs[start_idx:end_idx]
        
        # Convert to summary format
        items = []
        for run in page_runs:
            items.append(AgentRunSummaryResponse(
                id=run.id,
                query=run.query,
                agent_type=run.agent_type,
                status=run.status,
                started_at=run.started_at,
                completed_at=run.completed_at,
                final_decision=run.final_decision,
                confidence_score=run.confidence_score,
                total_steps=run.total_steps,
                execution_time_ms=run.execution_time_ms
            ))
        
        total_count = len(recent_runs)
        total_pages = (total_count + per_page - 1) // per_page
        
        return PaginatedResponse(
            items=items,
            total=total_count,
            page=page,
            per_page=per_page,
            pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list agent runs: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ================================
# SYSTEM MANAGEMENT ENDPOINTS
# ================================

@app.get("/api/v1/system/health", response_model=SystemHealth)
async def get_system_health() -> SystemHealth:
    """
    Get comprehensive system health information
    """
    try:
        # Get active executions
        active_executions = await execution_service.list_active_executions()
        
        # Get today's stats
        confidence_stats = await db_manager.get_confidence_statistics(days=1)
        
        # Calculate uptime
        uptime_seconds = 0
        if hasattr(app.state, 'startup_time'):
            uptime_seconds = int((datetime.utcnow() - app.state.startup_time).total_seconds())
        
        # Get recent runs for error rate
        recent_runs = await db_manager.get_recent_runs(limit=100)
        today_runs = [run for run in recent_runs if run.started_at.date() == datetime.utcnow().date()]
        
        error_rate = 0.0
        if today_runs:
            failed_runs = [run for run in today_runs if run.status == "failed"]
            error_rate = len(failed_runs) / len(today_runs)
        
        return SystemHealth(
            active_runs=len(active_executions),
            completed_runs_today=len(today_runs),
            avg_confidence_today=confidence_stats.get('avg_confidence'),
            error_rate_today=error_rate,
            uptime_seconds=uptime_seconds
        )
        
    except Exception as e:
        logger.error(f"Failed to get system health: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/v1/system/stats")
async def get_system_stats() -> Dict[str, Any]:
    """
    Get detailed system statistics
    """
    try:
        # Execution service stats
        active_executions = await execution_service.list_active_executions()
        
        # Database stats
        confidence_stats = await db_manager.get_confidence_statistics(days=7)
        
        # WebSocket stats
        ws_health = await get_websocket_health()
        
        # Agent registry stats
        available_agents = execution_service.agent_registry.list_agent_types()
        
        return {
            "execution_service": {
                "active_executions": len(active_executions),
                "max_concurrent": execution_service.max_concurrent_executions,
                "available_agents": available_agents
            },
            "database": confidence_stats,
            "websockets": ws_health,
            "system": {
                "environment": settings.app.ENVIRONMENT,
                "debug_mode": settings.app.DEBUG,
                "version": settings.app.APP_VERSION
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get system stats: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/v1/system/active-executions")
async def get_active_executions() -> List[Dict[str, Any]]:
    """
    Get list of currently active agent executions
    """
    try:
        return await execution_service.list_active_executions()
    except Exception as e:
        logger.error(f"Failed to get active executions: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ================================
# HEALTH CHECK & MONITORING
# ================================

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Basic health check endpoint
    """
    healthy = getattr(app.state, 'healthy', False)
    
    if healthy:
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": settings.app.APP_VERSION
        }
    else:
        raise HTTPException(status_code=503, detail="Service unavailable")


@app.get("/metrics")
async def get_metrics():
    """
    Prometheus metrics endpoint
    """
    try:
        if settings.monitoring.ENABLE_METRICS:
            return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
        else:
            raise HTTPException(status_code=404, detail="Metrics disabled")
    except Exception as e:
        logger.error(f"Failed to generate metrics: {e}")
        raise HTTPException(status_code=500, detail="Metrics unavailable")


# ================================
# ERROR HANDLERS
# ================================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content=create_error_response("Resource not found").dict()
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content=create_error_response("Internal server error").dict()
    )


# ================================
# BACKWARD COMPATIBILITY
# ================================

# Legacy endpoints for compatibility with your original code
@app.post("/init-agent-run/")
async def legacy_init_agent_run(request: dict) -> Dict[str, str]:
    """
    Legacy endpoint for backward compatibility
    """
    try:
        # Convert to new format
        agent_request = AgentRunRequest(
            query=request.get("query", ""),
            agent_type="refund_agent"
        )
        
        run_id = await execution_service.start_agent_execution(
            agent_type=agent_request.agent_type,
            query=agent_request.query
        )
        
        return {"run_id": run_id}
        
    except Exception as e:
        logger.error(f"Legacy endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get-next-step/{run_id}")
async def legacy_get_next_step(run_id: str) -> Dict[str, Any]:
    """
    Legacy endpoint for HTTP polling fallback
    """
    try:
        agent_run = await db_manager.get_agent_run(run_id)
        if not agent_run:
            return {"error": "Invalid run_id"}
        
        # Convert to legacy format
        steps = []
        if agent_run.steps:
            for step in agent_run.steps:
                steps.append({
                    "step": step.step_type,
                    "desc": step.description,
                    "obs": step.observation
                })
        
        return {
            "steps": steps,
            "done": agent_run.status in ["completed", "failed"],
            "final": agent_run.final_decision,
            "confidence": agent_run.confidence_score,
            "query": agent_run.query
        }
        
    except Exception as e:
        logger.error(f"Legacy endpoint error: {e}")
        return {"error": str(e)}


@app.get("/get-full-run/{run_id}")
async def legacy_get_full_run(run_id: str) -> Dict[str, Any]:
    """
    Legacy endpoint for full run data
    """
    try:
        response = await get_agent_run(run_id)
        
        # Convert to legacy format
        steps = []
        if response.steps:
            for step in response.steps:
                steps.append({
                    "step": step.step_type,
                    "desc": step.description,
                    "obs": step.observation
                })
        
        return {
            "final_decision": response.final_decision,
            "final_confidence": response.confidence_score,
            "done": response.is_completed,
            "steps": steps,
            "query": response.query
        }
        
    except HTTPException as e:
        return {"error": str(e.detail)}
    except Exception as e:
        logger.error(f"Legacy endpoint error: {e}")
        return {"error": str(e)}


# ================================
# APPLICATION ENTRY POINT
# ================================

if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "app.main:app",
        host=settings.app.HOST,
        port=settings.app.PORT,
        reload=settings.app.DEBUG,
        log_level=settings.monitoring.LOG_LEVEL.lower(),
        workers=1 if settings.app.DEBUG else settings.app.WORKERS
    )