# Database Migrations Guide

This guide explains how to work with Alembic for database migrations in the PortfolioPilot backend.

## Overview

Alembic is a lightweight database migration tool for SQLAlchemy and SQLModel. It allows you to:

- Track schema changes as SQL scripts
- Version control your database structure
- Roll back to previous versions if needed
- Generate migrations automatically from model changes

## Quick Start

### Prerequisites

1. Database is running:
   ```bash
   docker-compose up -d db
   ```

2. Dependencies are installed:
   ```bash
   cd backend && uv sync
   ```

### Check Current Migration Status

```bash
cd backend
uv run alembic current
```

This shows which migration is currently applied to the database.

### View Migration History

```bash
cd backend
uv run alembic history
```

Shows all migrations in order.

## Creating New Migrations

### Step 1: Define Your Models

Update your SQLModel models in `src/core/models/`:

```python
# src/core/models/user.py
from sqlmodel import SQLModel, Field

class User(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    email: str
    name: str
```

### Step 2: Import the Model

Add your model to `src/core/models/__init__.py`:

```python
from .user import User

__all__ = ["User"]
```

### Step 3: Generate Migration

Alembic will automatically detect changes to your models:

```bash
cd backend
uv run alembic revision --autogenerate -m "add users table"
```

This creates a new file in `alembic/versions/` with the SQL changes.

### Step 4: Review the Migration

Open the generated migration file and verify it looks correct:

```bash
# Review the migration
cat alembic/versions/XXX_add_users_table.py
```

### Step 5: Apply the Migration

```bash
cd backend
uv run alembic upgrade head
```

The `head` keyword applies all pending migrations up to the latest version.

## Common Migration Operations

### Apply All Pending Migrations

```bash
cd backend
uv run alembic upgrade head
```

### Apply a Specific Number of Migrations

```bash
cd backend
# Apply next single migration
uv run alembic upgrade +1

# Apply next two migrations
uv run alembic upgrade +2
```

### Revert to Previous Migration

```bash
cd backend
# Rollback one migration
uv run alembic downgrade -1

# Rollback to a specific revision
uv run alembic downgrade <revision_id>
```

### Check What Changes Will Be Applied

```bash
cd backend
# Show SQL that will be executed
uv run alembic upgrade head --sql
```

## Manual Migrations

If Alembic cannot automatically detect a change (e.g., custom SQL operations), you can create a manual migration:

```bash
cd backend
uv run alembic revision -m "custom operation"
```

This creates a migration file with empty `upgrade()` and `downgrade()` functions.

Edit the file to add your custom SQL:

```python
def upgrade() -> None:
    op.execute("CREATE INDEX idx_user_email ON user(email);")

def downgrade() -> None:
    op.execute("DROP INDEX idx_user_email;")
```

## Troubleshooting

### Migration Detection Issues

If Alembic doesn't detect your model changes:

1. Ensure models are imported in `alembic/env.py`
2. Check that `target_metadata` is set to `SQLModel.metadata`
3. Verify the model is registered (imported in `src/core/models/__init__.py`)

### Database Connection Issues

If migrations fail to connect to the database:

1. Verify database is running: `docker ps | grep app_db_local`
2. Check `.env` file has correct `DATABASE_URL`
3. Try connecting directly: `docker exec -it app_db_local psql -U admin -d app_db`

### Alembic Not Found

If you get "alembic: command not found":

```bash
cd backend
uv sync  # Reinstall dependencies
```

## Migration Workflow for Teams

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/add-portfolio-models
   ```

2. **Define your models** and generate migrations

3. **Commit both models and migrations**:
   ```bash
   git add src/core/models/ alembic/versions/
   git commit -m "Add portfolio models and migration"
   ```

4. **On main branch**, pull changes and apply:
   ```bash
   git pull origin main
   cd backend && uv run alembic upgrade head
   ```

## Initial Migration

The initial migration (`001_initial_pgvector_setup.py`) creates the pgvector extension for vector similarity search. This must be applied before using vector columns:

```bash
cd backend
uv run alembic upgrade head
```

## Environment Configuration

Migrations use the database URL from `.env`:

```
DATABASE_URL=postgresql+psycopg://admin:secret@localhost:5432/app_db
```

For production/different environments, ensure the correct `.env` file is loaded before running migrations.

## Docker Integration

After starting the database container, always apply migrations:

```bash
# Start database
docker-compose up -d db

# Apply migrations
cd backend && uv run alembic upgrade head
```

The `docker-compose.yml` includes comments with the migration workflow.

## Additional Resources

- [Alembic Documentation](https://alembic.sqlalchemy.org/)
- [SQLModel Documentation](https://sqlmodel.tiangolo.com/)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)

