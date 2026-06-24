"""add fallback_sql to thrive_agent_run

Revision ID: c233fb500001
Revises: b228fa11ed01
Create Date: 2026-06-24

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "c233fb500001"
down_revision: Union[str, Sequence[str], None] = "b228fa11ed01"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add the ``fallback_sql`` column to ``thrive_agent_run`` (Epic #228 / #233).

    When the agentic Vanna fallback fires, the runtime marks the agent run
    with ``status='fallback_invoked'`` and stashes the Vanna-generated SQL
    in this column. NULL on every existing row — they never invoked a
    fallback — so no backfill is needed.
    """
    with op.batch_alter_table("thrive_agent_run", schema=None) as batch_op:
        batch_op.add_column(sa.Column("fallback_sql", sa.Text(), nullable=True))


def downgrade() -> None:
    """Drop the ``fallback_sql`` column."""
    with op.batch_alter_table("thrive_agent_run", schema=None) as batch_op:
        batch_op.drop_column("fallback_sql")
