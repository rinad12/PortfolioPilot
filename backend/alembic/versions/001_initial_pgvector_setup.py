"""Initial migration: Create pgvector extension.

Revision ID: 001_initial_pgvector_setup
Revises: 
Create Date: 2025-11-23 00:00:00.000000

"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "001_initial_pgvector_setup"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create pgvector extension for vector similarity search."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")


def downgrade() -> None:
    """Drop pgvector extension (use with caution in production)."""
    op.execute("DROP EXTENSION IF NOT EXISTS vector CASCADE;")

