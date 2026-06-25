"""add show_thinking_process to thrive_user

Revision ID: f05a0e34dd21
Revises: c8d5e3a90212
Create Date: 2026-06-23 10:13:43.773482

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "f05a0e34dd21"
down_revision: Union[str, Sequence[str], None] = "c8d5e3a90212"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add the ``show_thinking_process`` user preference (Epic #222).

    Adds a nullable Boolean column via ``batch_alter_table`` (SQLite has no
    direct ``ALTER COLUMN``) and backfills every existing row to ``False``
    so the toggle defaults to hidden for the entire user base, matching the
    Epic's acceptance criteria.
    """
    with op.batch_alter_table("thrive_user", schema=None) as batch_op:
        batch_op.add_column(sa.Column("show_thinking_process", sa.Boolean(), nullable=True))
    op.execute("UPDATE thrive_user SET show_thinking_process = 0 WHERE show_thinking_process IS NULL")


def downgrade() -> None:
    """Drop the ``show_thinking_process`` column."""
    with op.batch_alter_table("thrive_user", schema=None) as batch_op:
        batch_op.drop_column("show_thinking_process")
