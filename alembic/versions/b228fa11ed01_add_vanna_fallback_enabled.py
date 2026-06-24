"""add vanna_fallback_enabled to thrive_user

Revision ID: b228fa11ed01
Revises: f05a0e34dd21
Create Date: 2026-06-23

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "b228fa11ed01"
down_revision: Union[str, Sequence[str], None] = "f05a0e34dd21"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add the ``vanna_fallback_enabled`` user preference (Epic #228 / #232).

    Per-user opt-in for the agentic-mode Vanna fallback. Combined with the
    per-deploy secrets flag ``[agent.fallback].enabled`` in
    ``agent.runtime._is_fallback_feature_enabled``; both must be True for
    the fallback path to fire. Default False on every row so the feature
    is dormant for the entire user base until each user explicitly opts in.
    """
    with op.batch_alter_table("thrive_user", schema=None) as batch_op:
        batch_op.add_column(sa.Column("vanna_fallback_enabled", sa.Boolean(), nullable=True))
    op.execute("UPDATE thrive_user SET vanna_fallback_enabled = 0 WHERE vanna_fallback_enabled IS NULL")


def downgrade() -> None:
    """Drop the ``vanna_fallback_enabled`` column."""
    with op.batch_alter_table("thrive_user", schema=None) as batch_op:
        batch_op.drop_column("vanna_fallback_enabled")
