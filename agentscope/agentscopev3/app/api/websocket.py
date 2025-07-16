"""
AgentScope WebSocket API

Real-time communication endpoints for streaming agent execution updates,
replacing HTTP polling with efficient WebSocket communication.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Set, Any
from uuid import UUID

from fastapi import WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.routing import APIRouter
from pydantic import BaseModel, ValidationError

from app.core.config import settings
from app.core.database import db_manager
from app.services.execution_service import execution_service
from app.models.schemas import (
    AgentRunResponse, StepStreamUpdate, RunStatusUpdate, 
    ConfidenceUpdate, agent_run_to_response
)

# Configure logging
logger = logging.getLogger(__name__)

# WebSocket router
websocket_router = APIRouter(prefix="/ws", tags=["websocket"])


class WebSocketMessage(BaseModel):
    """Base WebSocket message structure"""
    type: str
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime = None
    
    def __init__(self, **kwargs):
        if 'timestamp' not in kwargs:
            kwargs['timestamp'] = datetime.utcnow()
        super().__init__(**kwargs)


class ConnectionManager:
    """
    Manages WebSocket connections with room-based subscriptions
    """
    
    def __init__(self):
        # Active connections by run_id
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        
        # Connection metadata
        self.connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}
        
        # Global connections (for system-wide broadcasts)
        self.global_connections: Set[WebSocket] = set()
        
        # Performance tracking
        self.total_connections = 0
        self.messages_sent = 0
        self.connection_start_times: Dict[WebSocket, datetime] = {}
    
    async def connect(self, websocket: WebSocket, run_id: Optional[str] = None) -> None:
        """Accept a new WebSocket connection"""
        try:
            await websocket.accept()
            
            # Track connection
            self.total_connections += 1
            self.connection_start_times[websocket] = datetime.utcnow()
            
            # Store metadata
            self.connection_metadata[websocket] = {
                "connected_at": datetime.utcnow(),
                "run_id": run_id,
                "messages_sent": 0,
                "last_activity": datetime.utcnow()
            }
            
            if run_id:
                # Add to run-specific room
                if run_id not in self.active_connections:
                    self.active_connections[run_id] = set()
                self.active_connections[run_id].add(websocket)
                
                logger.info(f"WebSocket connected to run {run_id}")
            else:
                # Add to global connections
                self.global_connections.add(websocket)
                logger.info("WebSocket connected globally")
            
            # Send welcome message
            await self.send_personal_message(websocket, {
                "type": "connection_established",
                "data": {
                    "run_id": run_id,
                    "server_time": datetime.utcnow().isoformat(),
                    "capabilities": ["real_time_steps", "execution_status", "error_notifications"]
                }
            })
            
        except Exception as e:
            logger.error(f"Failed to establish WebSocket connection: {e}")
            raise
    
    async def disconnect(self, websocket: WebSocket) -> None:
        """Handle WebSocket disconnection"""
        try:
            # Remove from run-specific rooms
            for run_id, connections in self.active_connections.items():
                if websocket in connections:
                    connections.remove(websocket)
                    logger.info(f"WebSocket disconnected from run {run_id}")
                    
                    # Clean up empty rooms
                    if not connections:
                        del self.active_connections[run_id]
            
            # Remove from global connections
            if websocket in self.global_connections:
                self.global_connections.remove(websocket)
                logger.info("WebSocket disconnected globally")
            
            # Clean up metadata
            if websocket in self.connection_metadata:
                metadata = self.connection_metadata[websocket]
                connection_duration = datetime.utcnow() - metadata["connected_at"]
                logger.debug(f"Connection lasted {connection_duration.total_seconds():.1f}s, "
                           f"sent {metadata['messages_sent']} messages")
                del self.connection_metadata[websocket]
            
            if websocket in self.connection_start_times:
                del self.connection_start_times[websocket]
            
        except Exception as e:
            logger.error(f"Error during WebSocket disconnect: {e}")
    
    async def send_personal_message(self, websocket: WebSocket, message: Dict[str, Any]) -> bool:
        """Send message to a specific WebSocket connection"""
        try:
            # Create structured message
            ws_message = WebSocketMessage(**message)
            message_json = ws_message.json()
            
            await websocket.send_text(message_json)
            
            # Update metadata
            if websocket in self.connection_metadata:
                self.connection_metadata[websocket]["messages_sent"] += 1
                self.connection_metadata[websocket]["last_activity"] = datetime.utcnow()
            
            self.messages_sent += 1
            return True
            
        except WebSocketDisconnect:
            logger.debug("WebSocket disconnected during send")
            await self.disconnect(websocket)
            return False
        except Exception as e:
            logger.error(f"Failed to send WebSocket message: {e}")
            return False
    
    async def broadcast_to_run(self, run_id: str, message: Dict[str, Any]) -> int:
        """Broadcast message to all connections subscribed to a run"""
        if run_id not in self.active_connections:
            return 0
        
        connections = list(self.active_connections[run_id])  # Copy to avoid modification during iteration
        successful_sends = 0
        disconnected_connections = []
        
        for websocket in connections:
            success = await self.send_personal_message(websocket, message)
            if success:
                successful_sends += 1
            else:
                disconnected_connections.append(websocket)
        
        # Clean up disconnected connections
        for websocket in disconnected_connections:
            await self.disconnect(websocket)
        
        return successful_sends
    
    async def broadcast_global(self, message: Dict[str, Any]) -> int:
        """Broadcast message to all global connections"""
        connections = list(self.global_connections)  # Copy to avoid modification
        successful_sends = 0
        disconnected_connections = []
        
        for websocket in connections:
            success = await self.send_personal_message(websocket, message)
            if success:
                successful_sends += 1
            else:
                disconnected_connections.append(websocket)
        
        # Clean up disconnected connections
        for websocket in disconnected_connections:
            await self.disconnect(websocket)
        
        return successful_sends
    
    def get_connection_count(self, run_id: Optional[str] = None) -> int:
        """Get number of active connections"""
        if run_id:
            return len(self.active_connections.get(run_id, set()))
        else:
            total = len(self.global_connections)
            for connections in self.active_connections.values():
                total += len(connections)
            return total
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection manager statistics"""
        active_runs = len(self.active_connections)
        total_run_connections = sum(len(conns) for conns in self.active_connections.values())
        
        return {
            "total_connections_ever": self.total_connections,
            "active_global_connections": len(self.global_connections),
            "active_run_connections": total_run_connections,
            "active_runs_with_subscribers": active_runs,
            "total_messages_sent": self.messages_sent,
            "connection_rooms": {
                run_id: len(connections) 
                for run_id, connections in self.active_connections.items()
            }
        }


# Global connection manager
connection_manager = ConnectionManager()


@websocket_router.websocket("/runs/{run_id}")
async def websocket_run_endpoint(websocket: WebSocket, run_id: str):
    """
    WebSocket endpoint for real-time agent run updates
    
    Provides:
    - Real-time step updates as agent executes
    - Execution status changes
    - Confidence score updates
    - Error notifications
    - Completion notifications
    """
    await connection_manager.connect(websocket, run_id)
    
    try:
        # Check if run exists
        execution_status = await execution_service.get_execution_status(run_id)
        if not execution_status:
            await connection_manager.send_personal_message(websocket, {
                "type": "error",
                "data": {
                    "message": f"Run {run_id} not found",
                    "code": "RUN_NOT_FOUND"
                }
            })
            return
        
        # Send initial state
        await _send_initial_state(websocket, run_id)
        
        # Subscribe to execution service updates
        await execution_service.subscribe_to_execution(run_id, websocket)
        
        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Wait for messages from client with timeout
                message = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0  # 30 second timeout
                )
                
                # Process client message
                await _handle_client_message(websocket, run_id, message)
                
            except asyncio.TimeoutError:
                # Send heartbeat to keep connection alive
                await connection_manager.send_personal_message(websocket, {
                    "type": "heartbeat",
                    "data": {"server_time": datetime.utcnow().isoformat()}
                })
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for run {run_id}")
    except Exception as e:
        logger.error(f"WebSocket error for run {run_id}: {e}")
        try:
            await connection_manager.send_personal_message(websocket, {
                "type": "error",
                "data": {
                    "message": f"Server error: {str(e)}",
                    "code": "INTERNAL_ERROR"
                }
            })
        except:
            pass
    finally:
        await connection_manager.disconnect(websocket)


@websocket_router.websocket("/system")
async def websocket_system_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for system-wide updates
    
    Provides:
    - Active execution count
    - System health metrics
    - Global announcements
    """
    await connection_manager.connect(websocket)
    
    try:
        # Send initial system state
        await _send_system_state(websocket)
        
        # Keep connection alive
        while True:
            try:
                message = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=60.0  # 1 minute timeout for system connections
                )
                
                await _handle_system_message(websocket, message)
                
            except asyncio.TimeoutError:
                # Send system heartbeat
                await connection_manager.send_personal_message(websocket, {
                    "type": "system_heartbeat",
                    "data": await _get_system_stats()
                })
                
    except WebSocketDisconnect:
        logger.info("System WebSocket disconnected")
    except Exception as e:
        logger.error(f"System WebSocket error: {e}")
    finally:
        await connection_manager.disconnect(websocket)


async def _send_initial_state(websocket: WebSocket, run_id: str) -> None:
    """Send initial run state to newly connected client"""
    try:
        # Get current execution status
        execution_status = await execution_service.get_execution_status(run_id)
        
        # Get database state
        agent_run = await db_manager.get_agent_run(run_id)
        
        if agent_run:
            # Send complete current state
            response = agent_run_to_response(agent_run)
            await connection_manager.send_personal_message(websocket, {
                "type": "initial_state",
                "data": response.dict()
            })
        else:
            # Send execution status only
            await connection_manager.send_personal_message(websocket, {
                "type": "execution_status",
                "data": execution_status
            })
            
    except Exception as e:
        logger.error(f"Failed to send initial state for run {run_id}: {e}")


async def _send_system_state(websocket: WebSocket) -> None:
    """Send initial system state to newly connected client"""
    try:
        stats = await _get_system_stats()
        await connection_manager.send_personal_message(websocket, {
            "type": "system_state",
            "data": stats
        })
    except Exception as e:
        logger.error(f"Failed to send system state: {e}")


async def _get_system_stats() -> Dict[str, Any]:
    """Get current system statistics"""
    try:
        # Get execution service stats
        active_executions = await execution_service.list_active_executions()
        
        # Get connection manager stats
        connection_stats = connection_manager.get_stats()
        
        # Get database stats
        confidence_stats = await db_manager.get_confidence_statistics(days=1)
        
        return {
            "active_executions": len(active_executions),
            "connection_stats": connection_stats,
            "confidence_stats": confidence_stats,
            "server_time": datetime.utcnow().isoformat(),
            "uptime_info": {
                "service_initialized": execution_service._initialized,
                "total_connections": connection_stats["total_connections_ever"]
            }
        }
    except Exception as e:
        logger.error(f"Failed to get system stats: {e}")
        return {"error": "Failed to retrieve system stats"}


async def _handle_client_message(websocket: WebSocket, run_id: str, message: str) -> None:
    """Handle incoming message from client"""
    try:
        data = json.loads(message)
        message_type = data.get("type")
        
        if message_type == "ping":
            # Respond to ping
            await connection_manager.send_personal_message(websocket, {
                "type": "pong",
                "data": {"original_timestamp": data.get("timestamp")}
            })
            
        elif message_type == "request_status":
            # Send current status
            execution_status = await execution_service.get_execution_status(run_id)
            await connection_manager.send_personal_message(websocket, {
                "type": "status_response",
                "data": execution_status
            })
            
        elif message_type == "cancel_execution":
            # Cancel execution (if authorized)
            success = await execution_service.cancel_execution(run_id)
            await connection_manager.send_personal_message(websocket, {
                "type": "cancel_response",
                "data": {"success": success, "run_id": run_id}
            })
            
        else:
            logger.warning(f"Unknown message type from client: {message_type}")
            
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON from client: {message}")
    except Exception as e:
        logger.error(f"Error handling client message: {e}")


async def _handle_system_message(websocket: WebSocket, message: str) -> None:
    """Handle incoming system message from client"""
    try:
        data = json.loads(message)
        message_type = data.get("type")
        
        if message_type == "get_stats":
            stats = await _get_system_stats()
            await connection_manager.send_personal_message(websocket, {
                "type": "stats_response",
                "data": stats
            })
            
        elif message_type == "list_executions":
            executions = await execution_service.list_active_executions()
            await connection_manager.send_personal_message(websocket, {
                "type": "executions_list",
                "data": executions
            })
            
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON from system client: {message}")
    except Exception as e:
        logger.error(f"Error handling system message: {e}")


# Utility functions for broadcasting system-wide events

async def broadcast_system_event(event_type: str, data: Dict[str, Any]) -> int:
    """Broadcast an event to all system WebSocket connections"""
    return await connection_manager.broadcast_global({
        "type": event_type,
        "data": data
    })


async def broadcast_run_event(run_id: str, event_type: str, data: Dict[str, Any]) -> int:
    """Broadcast an event to all connections subscribed to a specific run"""
    return await connection_manager.broadcast_to_run(run_id, {
        "type": event_type,
        "data": data
    })


async def notify_execution_started(run_id: str, agent_type: str, query: str) -> None:
    """Notify about new execution start"""
    await broadcast_system_event("execution_started", {
        "run_id": run_id,
        "agent_type": agent_type,
        "query": query[:100] + "..." if len(query) > 100 else query,
        "started_at": datetime.utcnow().isoformat()
    })


async def notify_execution_completed(run_id: str, final_result: str, confidence: float) -> None:
    """Notify about execution completion"""
    await broadcast_system_event("execution_completed", {
        "run_id": run_id,
        "final_result": final_result,
        "confidence": confidence,
        "completed_at": datetime.utcnow().isoformat()
    })


# Health check endpoint for WebSocket service
async def get_websocket_health() -> Dict[str, Any]:
    """Get WebSocket service health information"""
    stats = connection_manager.get_stats()
    return {
        "status": "healthy",
        "active_connections": stats["active_global_connections"] + stats["active_run_connections"],
        "total_messages_sent": stats["total_messages_sent"],
        "active_runs_with_subscribers": stats["active_runs_with_subscribers"],
        "timestamp": datetime.utcnow().isoformat()
    }