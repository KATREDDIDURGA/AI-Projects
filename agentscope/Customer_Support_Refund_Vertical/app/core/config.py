"""
AgentScope Configuration Management

Centralized configuration using Pydantic Settings for type safety,
environment-based configuration, and secret management.
"""

import os
import secrets
from typing import Any, Dict, List, Optional, Union
from pydantic import AnyHttpUrl, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """Database configuration settings"""
    
    # SQLite Configuration (No installation required)
    DB_PATH: str = "data/agentscope.db"
    DB_BACKUP_PATH: str = "data/backups"
    
    # Connection Pool Settings (SQLite specific)
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    DB_POOL_TIMEOUT: int = 30
    DB_POOL_RECYCLE: int = 3600
    
    # SQLite-specific settings
    DB_ECHO_SQL: bool = False  # Set to True for SQL debugging
    DB_WAL_MODE: bool = True   # Enable WAL mode for better concurrency
    DB_FOREIGN_KEYS: bool = True
    DB_TIMEOUT: int = 20       # SQLite connection timeout
    
    @validator("DB_PATH", pre=True)
    def validate_db_path(cls, v: str) -> str:
        # Ensure the directory exists
        db_dir = os.path.dirname(v)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        return v
    
    @property
    def database_url(self) -> str:
        """Construct SQLite connection URL for async operations"""
        # SQLite async URL format
        return f"sqlite+aiosqlite:///{self.DB_PATH}"
    
    @property
    def sync_database_url(self) -> str:
        """Sync SQLite URL for migrations and sync operations"""
        return f"sqlite:///{self.DB_PATH}"
    
    @property
    def backup_enabled(self) -> bool:
        """Check if backup directory is configured"""
        return bool(self.DB_BACKUP_PATH)


class RedisSettings(BaseSettings):
    """Redis configuration for real-time state management"""
    
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: Optional[str] = None
    REDIS_DB: int = 0
    REDIS_SSL: bool = False
    
    # Fallback option: Use in-memory storage if Redis not available
    USE_REDIS: bool = True
    FALLBACK_TO_MEMORY: bool = True
    
    # Connection Pool Settings
    REDIS_MAX_CONNECTIONS: int = 20
    REDIS_SOCKET_TIMEOUT: int = 5
    REDIS_SOCKET_CONNECT_TIMEOUT: int = 5
    
    # AgentScope-specific Redis keys
    AGENT_RUNS_PREFIX: str = "agentscope:runs"
    AGENT_METRICS_PREFIX: str = "agentscope:metrics"
    WEBSOCKET_ROOMS_PREFIX: str = "agentscope:ws"
    
    @property
    def redis_url(self) -> str:
        """Construct Redis connection URL"""
        if not self.USE_REDIS:
            return "memory://"  # Fallback to in-memory
        
        auth = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
        protocol = "rediss" if self.REDIS_SSL else "redis"
        return f"{protocol}://{auth}{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"


class AIServiceSettings(BaseSettings):
    """AI service configuration"""
    
    # Together AI Configuration
    TOGETHER_API_KEY: str = "dummy_key_for_development"  # Default for development
    TOGETHER_MODEL: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    TOGETHER_MAX_TOKENS: int = 1024
    TOGETHER_TEMPERATURE: float = 0.1
    TOGETHER_TIMEOUT: int = 30
    
    # OpenAI Compatibility (future-proofing)
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4"
    
    # Rate Limiting
    AI_REQUESTS_PER_MINUTE: int = 60
    AI_REQUESTS_PER_HOUR: int = 1000
    
    # Retry Configuration
    AI_MAX_RETRIES: int = 3
    AI_RETRY_DELAY: float = 1.0
    AI_BACKOFF_MULTIPLIER: float = 2.0
    
    @validator("TOGETHER_API_KEY", pre=True)
    def validate_together_key(cls, v: str) -> str:
        if not v or v == "dummy_key_for_development":
            print("⚠️  WARNING: Using dummy API key. Set TOGETHER_API_KEY in .env for actual AI calls")
        return v


class AgentSettings(BaseSettings):
    """Agent-specific configuration"""
    
    # Confidence Scoring
    MIN_CONFIDENCE_THRESHOLD: float = 0.3
    HIGH_CONFIDENCE_THRESHOLD: float = 0.8
    FALLBACK_CONFIDENCE_THRESHOLD: float = 0.2
    
    # Execution Timeouts
    AGENT_STEP_TIMEOUT: int = 30  # seconds per step
    AGENT_TOTAL_TIMEOUT: int = 300  # total agent execution timeout
    MAX_AGENT_STEPS: int = 20
    
    # Policy & Transaction Lookup
    TRANSACTION_LOOKUP_TIMEOUT: int = 5
    POLICY_LOOKUP_TIMEOUT: int = 5
    
    # Confidence Calculation Weights
    POLICY_EXISTS_WEIGHT: float = 0.3
    REFUND_ALLOWED_WEIGHT: float = 0.4
    TIMEFRAME_COMPLIANCE_WEIGHT: float = 0.3
    
    @validator("MIN_CONFIDENCE_THRESHOLD", "HIGH_CONFIDENCE_THRESHOLD", "FALLBACK_CONFIDENCE_THRESHOLD")
    def validate_confidence_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence thresholds must be between 0.0 and 1.0")
        return v


class MonitoringSettings(BaseSettings):
    """Monitoring and observability configuration"""
    
    # Prometheus Metrics
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    METRICS_PATH: str = "/metrics"
    
    # Sentry Error Tracking
    SENTRY_DSN: Optional[str] = None
    SENTRY_ENVIRONMENT: str = "development"
    SENTRY_TRACES_SAMPLE_RATE: float = 0.1
    
    # Structured Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"  # json or console
    LOG_FILE: Optional[str] = None
    
    # Performance Monitoring
    ENABLE_PROFILING: bool = False
    SLOW_QUERY_THRESHOLD: float = 1.0  # seconds
    
    @validator("LOG_LEVEL")
    def validate_log_level(cls, v: str) -> str:
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of {valid_levels}")
        return v.upper()


class SecuritySettings(BaseSettings):
    """Security configuration"""
    
    # JWT Configuration
    SECRET_KEY: str = secrets.token_urlsafe(32)
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CORS Settings
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = []
    
    # API Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 100
    RATE_LIMIT_BURST: int = 20
    
    # WebSocket Security
    WEBSOCKET_MAX_CONNECTIONS: int = 100
    WEBSOCKET_MESSAGE_SIZE_LIMIT: int = 1024 * 1024  # 1MB
    
    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)


class AppSettings(BaseSettings):
    """Main application settings"""
    
    # Application Info
    APP_NAME: str = "AgentScope"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = "AI Agent Timeline Debugger & Reliability Monitor"
    
    # Environment
    ENVIRONMENT: str = "development"  # development, staging, production
    DEBUG: bool = True
    
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1
    
    # Data Directories
    DATA_DIR: str = "data"
    LOGS_DIR: str = "logs"
    TEMP_DIR: str = "/tmp/agentscope"
    
    # File Upload Limits
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_FILE_TYPES: List[str] = [".csv", ".json", ".xlsx"]
    
    @validator("ENVIRONMENT")
    def validate_environment(cls, v: str) -> str:
        valid_envs = ["development", "staging", "production"]
        if v not in valid_envs:
            raise ValueError(f"ENVIRONMENT must be one of {valid_envs}")
        return v
    
    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"
    
    @property
    def is_development(self) -> bool:
        return self.ENVIRONMENT == "development"


class Settings(BaseSettings):
    """Main settings class that combines all configuration sections"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"  # Ignore unknown environment variables
    )
    
    # Application Settings
    app: AppSettings = AppSettings()
    
    # Infrastructure Settings
    database: DatabaseSettings = DatabaseSettings()
    redis: RedisSettings = RedisSettings()
    
    # AI & Agent Settings
    ai: AIServiceSettings = AIServiceSettings()
    agents: AgentSettings = AgentSettings()
    
    # Monitoring & Security
    monitoring: MonitoringSettings = MonitoringSettings()
    security: SecuritySettings = SecuritySettings()
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ensure_directories()
        self._validate_configuration()
    
    def _ensure_directories(self) -> None:
        """Create necessary directories if they don't exist"""
        for directory in [self.app.DATA_DIR, self.app.LOGS_DIR, self.app.TEMP_DIR]:
            os.makedirs(directory, exist_ok=True)
    
    def _validate_configuration(self) -> None:
        """Validate cross-setting dependencies"""
        if self.app.is_production:
            if self.app.DEBUG:
                raise ValueError("DEBUG must be False in production")
            if not self.monitoring.SENTRY_DSN:
                print("WARNING: SENTRY_DSN not set in production")
    
    @property
    def redis_key_agent_run(self) -> str:
        """Generate Redis key for agent run storage"""
        def get_key(run_id: str) -> str:
            return f"{self.redis.AGENT_RUNS_PREFIX}:{run_id}"
        return get_key
    
    @property
    def redis_key_websocket_room(self) -> str:
        """Generate Redis key for WebSocket room management"""
        def get_key(room_id: str) -> str:
            return f"{self.redis.WEBSOCKET_ROOMS_PREFIX}:{room_id}"
        return get_key


# Global settings instance
settings = Settings()


# Environment-specific configuration helpers
def get_database_url() -> str:
    """Get database URL for current environment"""
    return settings.database.database_url


def get_redis_url() -> str:
    """Get Redis URL for current environment"""
    return settings.redis.redis_url


def is_production() -> bool:
    """Check if running in production environment"""
    return settings.app.is_production


def get_ai_config() -> Dict[str, Any]:
    """Get AI service configuration"""
    return {
        "api_key": settings.ai.TOGETHER_API_KEY,
        "model": settings.ai.TOGETHER_MODEL,
        "max_tokens": settings.ai.TOGETHER_MAX_TOKENS,
        "temperature": settings.ai.TOGETHER_TEMPERATURE,
        "timeout": settings.ai.TOGETHER_TIMEOUT
    }