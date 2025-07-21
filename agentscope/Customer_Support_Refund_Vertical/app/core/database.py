"""
AgentScope Database Management

Async SQLite database connection management with proper schema,
connection pooling, and transaction handling.
"""

import asyncio
import logging
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator, Dict, List, Optional, Any
from pathlib import Path

from sqlalchemy import (
    Column, String, Integer, Float, DateTime, Text, Boolean, 
    ForeignKey, Index, event, pool
)
from sqlalchemy.ext.asyncio import (
    AsyncSession, async_sessionmaker, create_async_engine, AsyncEngine
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from app.core.config import settings

# Configure logging
logger = logging.getLogger(__name__)

# Base class for all database models
Base = declarative_base()


class AgentRun(Base):
    """Agent execution run record"""
    __tablename__ = "agent_runs"
    
    # Primary identification
    id = Column(String, primary_key=True)  # UUID run_id
    query = Column(Text, nullable=False)   # User query
    agent_type = Column(String, nullable=False, default="refund_agent")
    
    # Execution metadata
    status = Column(String, nullable=False, default="running")  # running, completed, failed
    started_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    
    # Results
    final_decision = Column(Text, nullable=True)
    confidence_score = Column(Float, nullable=True)
    
    # Error tracking
    error_message = Column(Text, nullable=True)
    error_step = Column(Integer, nullable=True)
    
    # Performance metrics
    total_steps = Column(Integer, default=0)
    execution_time_ms = Column(Integer, nullable=True)
    
    # Relationships
    steps = relationship("AgentStep", back_populates="run", cascade="all, delete-orphan")
    metrics = relationship("AgentMetric", back_populates="run", cascade="all, delete-orphan")
    
    # Indexes for better query performance
    __table_args__ = (
        Index('idx_agent_runs_status', 'status'),
        Index('idx_agent_runs_started_at', 'started_at'),
        Index('idx_agent_runs_agent_type', 'agent_type'),
    )


class AgentStep(Base):
    """Individual step in agent execution"""
    __tablename__ = "agent_steps"
    
    # Primary identification
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, ForeignKey("agent_runs.id"), nullable=False)
    step_number = Column(Integer, nullable=False)
    
    # Step details
    step_type = Column(String, nullable=False)  # Thought, Observation, Decision, Fallback
    description = Column(Text, nullable=False)
    observation = Column(Text, nullable=True)   # JSON serialized observation data
    
    # Timing
    started_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    duration_ms = Column(Integer, nullable=True)
    
    # Step-specific metrics
    confidence = Column(Float, nullable=True)
    success = Column(Boolean, nullable=True)
    
    # Relationships
    run = relationship("AgentRun", back_populates="steps")
    
    # Indexes
    __table_args__ = (
        Index('idx_agent_steps_run_id', 'run_id'),
        Index('idx_agent_steps_step_number', 'run_id', 'step_number'),
        Index('idx_agent_steps_type', 'step_type'),
    )


class AgentMetric(Base):
    """Performance and reliability metrics"""
    __tablename__ = "agent_metrics"
    
    # Primary identification
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, ForeignKey("agent_runs.id"), nullable=False)
    
    # Metric details
    metric_name = Column(String, nullable=False)   # confidence, execution_time, step_count, etc.
    metric_value = Column(Float, nullable=False)
    metric_unit = Column(String, nullable=True)    # seconds, percentage, count
    
    # Context
    step_number = Column(Integer, nullable=True)   # If metric is step-specific
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Additional metadata (renamed from 'metadata' to avoid SQLAlchemy conflict)
    extra_data = Column(Text, nullable=True)         # JSON for extra context
    
    # Relationships
    run = relationship("AgentRun", back_populates="metrics")
    
    # Indexes
    __table_args__ = (
        Index('idx_agent_metrics_run_id', 'run_id'),
        Index('idx_agent_metrics_name', 'metric_name'),
        Index('idx_agent_metrics_timestamp', 'timestamp'),
    )


class DatabaseManager:
    """Manages database connections and operations"""
    
    def __init__(self):
        self.engine: Optional[AsyncEngine] = None
        self.async_session_maker: Optional[async_sessionmaker] = None
        self.sync_engine = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize database connections and create tables"""
        if self._initialized:
            return
        
        try:
            # Create database directory if it doesn't exist
            db_path = Path(settings.database.DB_PATH)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create async engine with SQLite optimizations
            self.engine = create_async_engine(
                settings.database.database_url,
                poolclass=StaticPool,
                connect_args={
                    "check_same_thread": False,
                    "timeout": settings.database.DB_TIMEOUT,
                },
                echo=settings.database.DB_ECHO_SQL,
                future=True,
            )
            
            # Create sync engine for migrations and setup
            self.sync_engine = create_engine(
                settings.database.sync_database_url,
                poolclass=StaticPool,
                connect_args={
                    "check_same_thread": False,
                    "timeout": settings.database.DB_TIMEOUT,
                },
                echo=settings.database.DB_ECHO_SQL,
            )
            
            # Configure SQLite for better performance
            self._configure_sqlite()
            
            # Create tables
            await self._create_tables()
            
            # Create async session maker
            self.async_session_maker = async_sessionmaker(
                bind=self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )
            
            self._initialized = True
            logger.info(f"Database initialized successfully at {settings.database.DB_PATH}")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def _configure_sqlite(self) -> None:
        """Configure SQLite for optimal performance"""
        
        @event.listens_for(self.sync_engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            if isinstance(dbapi_connection, sqlite3.Connection):
                cursor = dbapi_connection.cursor()
                
                # Enable WAL mode for better concurrency
                if settings.database.DB_WAL_MODE:
                    cursor.execute("PRAGMA journal_mode=WAL")
                
                # Enable foreign key constraints
                if settings.database.DB_FOREIGN_KEYS:
                    cursor.execute("PRAGMA foreign_keys=ON")
                
                # Performance optimizations
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.execute("PRAGMA cache_size=10000")
                cursor.execute("PRAGMA temp_store=MEMORY")
                cursor.execute("PRAGMA mmap_size=268435456")  # 256MB
                
                cursor.close()
        
        # Configure async engine too
        @event.listens_for(self.engine.sync_engine, "connect")
        def set_sqlite_pragma_async(dbapi_connection, connection_record):
            if isinstance(dbapi_connection, sqlite3.Connection):
                cursor = dbapi_connection.cursor()
                
                if settings.database.DB_WAL_MODE:
                    cursor.execute("PRAGMA journal_mode=WAL")
                if settings.database.DB_FOREIGN_KEYS:
                    cursor.execute("PRAGMA foreign_keys=ON")
                
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.execute("PRAGMA cache_size=10000")
                cursor.execute("PRAGMA temp_store=MEMORY")
                cursor.execute("PRAGMA mmap_size=268435456")
                
                cursor.close()
    
    async def _create_tables(self) -> None:
        """Create database tables"""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    async def close(self) -> None:
        """Close database connections"""
        if self.engine:
            await self.engine.dispose()
        if self.sync_engine:
            self.sync_engine.dispose()
        self._initialized = False
        logger.info("Database connections closed")
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get async database session with automatic cleanup"""
        if not self._initialized:
            await self.initialize()
        
        async with self.async_session_maker() as session:
            try:
                yield session
                await session.commit()
            except Exception as e:
                await session.rollback()
                logger.error(f"Database transaction failed: {e}")
                raise
            finally:
                await session.close()
    
    async def create_agent_run(
        self, 
        run_id: str, 
        query: str, 
        agent_type: str = "refund_agent"
    ) -> AgentRun:
        """Create a new agent run record"""
        async with self.get_session() as session:
            agent_run = AgentRun(
                id=run_id,
                query=query,
                agent_type=agent_type,
                status="running",
                started_at=datetime.utcnow()
            )
            session.add(agent_run)
            await session.flush()  # Get the ID without committing
            return agent_run
    
    async def add_agent_step(
        self,
        run_id: str,
        step_number: int,
        step_type: str,
        description: str,
        observation: Optional[str] = None,
        confidence: Optional[float] = None
    ) -> AgentStep:
        """Add a step to an agent run"""
        async with self.get_session() as session:
            step = AgentStep(
                run_id=run_id,
                step_number=step_number,
                step_type=step_type,
                description=description,
                observation=observation,
                confidence=confidence,
                started_at=datetime.utcnow()
            )
            session.add(step)
            await session.flush()
            return step
    
    async def complete_agent_run(
        self,
        run_id: str,
        final_decision: str,
        confidence_score: float,
        total_steps: int,
        execution_time_ms: int,
        status: str = "completed"
    ) -> None:
        """Mark agent run as completed with results"""
        async with self.get_session() as session:
            result = await session.get(AgentRun, run_id)
            if result:
                result.status = status
                result.final_decision = final_decision
                result.confidence_score = confidence_score
                result.total_steps = total_steps
                result.execution_time_ms = execution_time_ms
                result.completed_at = datetime.utcnow()
    
    async def mark_agent_run_failed(
        self,
        run_id: str,
        error_message: str,
        error_step: Optional[int] = None
    ) -> None:
        """Mark agent run as failed"""
        async with self.get_session() as session:
            result = await session.get(AgentRun, run_id)
            if result:
                result.status = "failed"
                result.error_message = error_message
                result.error_step = error_step
                result.completed_at = datetime.utcnow()
    
    async def get_agent_run(self, run_id: str) -> Optional[AgentRun]:
        """Get agent run with all steps and metrics"""
        async with self.get_session() as session:
            # Using selectinload to eagerly load relationships
            from sqlalchemy.orm import selectinload
            from sqlalchemy import select
            
            stmt = select(AgentRun).options(
                selectinload(AgentRun.steps),
                selectinload(AgentRun.metrics)
            ).where(AgentRun.id == run_id)
            
            result = await session.execute(stmt)
            return result.scalar_one_or_none()
    
    async def get_recent_runs(self, limit: int = 50) -> List[AgentRun]:
        """Get recent agent runs"""
        async with self.get_session() as session:
            from sqlalchemy import select, desc
            
            stmt = select(AgentRun).order_by(desc(AgentRun.started_at)).limit(limit)
            result = await session.execute(stmt)
            return result.scalars().all()
    
    async def add_metric(
        self,
        run_id: str,
        metric_name: str,
        metric_value: float,
        metric_unit: Optional[str] = None,
        step_number: Optional[int] = None,
        extra_data: Optional[str] = None
    ) -> AgentMetric:
        """Add a performance metric"""
        async with self.get_session() as session:
            metric = AgentMetric(
                run_id=run_id,
                metric_name=metric_name,
                metric_value=metric_value,
                metric_unit=metric_unit,
                step_number=step_number,
                extra_data=extra_data,
                timestamp=datetime.utcnow()
            )
            session.add(metric)
            await session.flush()
            return metric
    
    async def get_confidence_statistics(self, days: int = 30) -> Dict[str, float]:
        """Get confidence score statistics for recent runs"""
        async with self.get_session() as session:
            from sqlalchemy import select, func
            from datetime import timedelta
            
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            stmt = select(
                func.avg(AgentRun.confidence_score).label('avg_confidence'),
                func.min(AgentRun.confidence_score).label('min_confidence'),
                func.max(AgentRun.confidence_score).label('max_confidence'),
                func.count(AgentRun.id).label('total_runs')
            ).where(
                AgentRun.started_at >= cutoff_date,
                AgentRun.confidence_score.isnot(None)
            )
            
            result = await session.execute(stmt)
            row = result.first()
            
            return {
                'avg_confidence': float(row.avg_confidence or 0),
                'min_confidence': float(row.min_confidence or 0),
                'max_confidence': float(row.max_confidence or 0),
                'total_runs': int(row.total_runs or 0)
            }


# Global database manager instance
db_manager = DatabaseManager()


# Dependency injection for FastAPI
async def get_database() -> AsyncGenerator[DatabaseManager, None]:
    """Dependency to get database manager"""
    if not db_manager._initialized:
        await db_manager.initialize()
    yield db_manager


# Utility functions for easy access
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Get database session for direct SQLAlchemy operations"""
    async with db_manager.get_session() as session:
        yield session