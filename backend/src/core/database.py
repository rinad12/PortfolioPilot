"""Database configuration and session management using SQLModel."""

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel

from core.config import get_settings

# Get settings
settings = get_settings()

# Create async engine
engine = create_async_engine(
    settings.get_database_url(),
    echo=False,
    future=True,
    pool_pre_ping=True,
)

# Session factory
async_session = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Get async database session for dependency injection."""
    async with async_session() as session:
        yield session


async def create_db_and_tables() -> None:
    """Create database tables from SQLModel metadata."""
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)


__all__ = ["SQLModel", "engine", "async_session", "get_session", "create_db_and_tables"]
