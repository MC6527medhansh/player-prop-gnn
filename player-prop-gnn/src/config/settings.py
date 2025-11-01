"""
Configuration settings for the Player Prop GNN project.
Loads from environment variables with sensible defaults.
"""
from pydantic_settings import BaseSettings
from pydantic import Field, validator
from typing import Optional, List
import os
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Project Paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    MODEL_DIR: str = "models/"
    DATA_DIR: str = "data/"
    LOG_DIR: str = "logs/"
    
    # Database Configuration
    DATABASE_URL: str = Field(
        default="postgresql://postgres:password@localhost:5432/football_props",
        description="PostgreSQL connection string"
    )
    DATABASE_HOST: str = "localhost"
    DATABASE_PORT: int = 5432
    DATABASE_NAME: str = "football_props"
    DATABASE_USER: str = "postgres"
    DATABASE_PASSWORD: str = "password"
    
    # Redis Configuration
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 4
    API_RELOAD: bool = False
    
    # Training Configuration
    RANDOM_SEED: int = 42
    TEST_SIZE: float = 0.2
    VALIDATION_SIZE: float = 0.1
    BATCH_SIZE: int = 32
    
    # Feature Engineering
    MIN_MATCHES_PER_PLAYER: int = 5
    LOOKBACK_WINDOW_MATCHES: int = 10
    
    # Bayesian Model Hyperparameters
    BAYESIAN_DRAWS: int = 2000
    BAYESIAN_TUNE: int = 1000
    BAYESIAN_CHAINS: int = 4
    BAYESIAN_TARGET_ACCEPT: float = 0.95
    
    # GNN Hyperparameters
    GNN_HIDDEN_DIM: int = 64
    GNN_NUM_LAYERS: int = 3
    GNN_ATTENTION_HEADS: int = 8
    GNN_DROPOUT: float = 0.2
    LEARNING_RATE: float = 0.001
    GNN_EPOCHS: int = 100
    
    # Caching
    CACHE_TTL_SECONDS: int = 3600
    ENABLE_CACHE: bool = True
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"  # or "text"
    
    # Data Collection
    FBREF_DELAY_SECONDS: int = 2
    MAX_RETRIES: int = 3
    TIMEOUT_SECONDS: int = 30
    
    # Monitoring
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    
    # Environment
    DEBUG: bool = False
    TESTING: bool = False
    ENVIRONMENT: str = "development"  # development, staging, production
    
    # API Keys (optional, for future use)
    SPORTS_API_KEY: Optional[str] = None
    ODDS_API_KEY: Optional[str] = None
    
    @validator("TEST_SIZE", "VALIDATION_SIZE")
    def validate_split_size(cls, v):
        """Ensure split sizes are between 0 and 1."""
        if not 0 < v < 1:
            raise ValueError("Split size must be between 0 and 1")
        return v
    
    @validator("LOG_LEVEL")
    def validate_log_level(cls, v):
        """Ensure log level is valid."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()
    
    @validator("ENVIRONMENT")
    def validate_environment(cls, v):
        """Ensure environment is valid."""
        valid_envs = ["development", "staging", "production"]
        if v.lower() not in valid_envs:
            raise ValueError(f"Environment must be one of {valid_envs}")
        return v.lower()
    
    @property
    def database_url_async(self) -> str:
        """Get async database URL for SQLAlchemy."""
        return self.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")
    
    @property
    def redis_url(self) -> str:
        """Get Redis connection URL."""
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
    
    @property
    def model_path(self) -> Path:
        """Get absolute path to models directory."""
        return self.PROJECT_ROOT / self.MODEL_DIR
    
    @property
    def data_path(self) -> Path:
        """Get absolute path to data directory."""
        return self.PROJECT_ROOT / self.DATA_DIR
    
    @property
    def log_path(self) -> Path:
        """Get absolute path to logs directory."""
        return self.PROJECT_ROOT / self.LOG_DIR
    
    def ensure_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.model_path,
            self.data_path,
            self.log_path,
            self.data_path / "raw",
            self.data_path / "processed",
            self.data_path / "external",
            self.data_path / "schemas",
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            # Create .gitkeep files to preserve empty directories in git
            gitkeep = directory / ".gitkeep"
            if not gitkeep.exists() and directory.name in ["raw", "processed", "external"]:
                gitkeep.touch()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
settings = Settings()

# Ensure directories exist on import
settings.ensure_directories()


# Convenience functions for common configurations
def get_postgres_connection_dict() -> dict:
    """Get PostgreSQL connection parameters as dict."""
    return {
        "host": settings.DATABASE_HOST,
        "port": settings.DATABASE_PORT,
        "database": settings.DATABASE_NAME,
        "user": settings.DATABASE_USER,
        "password": settings.DATABASE_PASSWORD,
    }


def get_redis_connection_dict() -> dict:
    """Get Redis connection parameters as dict."""
    config = {
        "host": settings.REDIS_HOST,
        "port": settings.REDIS_PORT,
        "db": settings.REDIS_DB,
        "decode_responses": True,
    }
    if settings.REDIS_PASSWORD:
        config["password"] = settings.REDIS_PASSWORD
    return config


def is_production() -> bool:
    """Check if running in production environment."""
    return settings.ENVIRONMENT == "production"


def is_development() -> bool:
    """Check if running in development environment."""
    return settings.ENVIRONMENT == "development"


# Example usage:
if __name__ == "__main__":
    print("=== Player Prop GNN Configuration ===")
    print(f"Environment: {settings.ENVIRONMENT}")
    print(f"Database URL: {settings.DATABASE_URL}")
    print(f"Redis URL: {settings.redis_url}")
    print(f"Model Directory: {settings.model_path}")
    print(f"Data Directory: {settings.data_path}")
    print(f"Debug Mode: {settings.DEBUG}")
    print(f"API Port: {settings.API_PORT}")
    print(f"Random Seed: {settings.RANDOM_SEED}")
