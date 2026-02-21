"""Database configuration using Pydantic Settings."""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings

# config.py -> core/ -> src/ -> backend/ -> project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database Configuration
    postgres_user: str = "admin"
    postgres_password: str = "secret"
    postgres_db: str = "app_db"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    database_url: Optional[str] = None

    class Config:
        env_file = str(_PROJECT_ROOT / ".env")
        case_sensitive = False

    def get_database_url(self) -> str:
        """Get the database URL from settings or construct it."""
        if self.database_url:
            return self.database_url
        return (
            f"postgresql+psycopg://{self.postgres_user}:"
            f"{self.postgres_password}@{self.postgres_host}:"
            f"{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
