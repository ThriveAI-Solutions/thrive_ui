"""add index on AgentRun.user_message_id

Adds an index ``ix_thrive_agent_run_user_message_id`` on
``thrive_agent_run.user_message_id`` so the LEFT JOIN added by the
Question Audit Scope filter (Epic #166 / Feature #167) stays cheap as
audit data grows. The audit base query joins
``AgentRun.user_message_id = Message.id`` on every audit page render and
on every CSV export; without the index this is a full-scan of
``thrive_agent_run``.

The index is hygiene; safe to leave in place on rollback. Migration is
symmetric (create / drop).

Revision ID: 188ab391e291
Revises: b48e797bfa81
Create Date: 2026-06-11 09:35:57.299037

"""

from typing import Sequence, Union

from alembic import op
from sqlalchemy import inspect

# revision identifiers, used by Alembic.
revision: str = "188ab391e291"
down_revision: Union[str, Sequence[str], None] = "b48e797bfa81"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_INDEX_NAME = "ix_thrive_agent_run_user_message_id"
_TABLE_NAME = "thrive_agent_run"
_COLUMN_NAME = "user_message_id"


def upgrade() -> None:
    """Upgrade schema."""
    bind = op.get_bind()
    inspector = inspect(bind)
    existing = {ix["name"] for ix in inspector.get_indexes(_TABLE_NAME)}
    if _INDEX_NAME not in existing:
        op.create_index(_INDEX_NAME, _TABLE_NAME, [_COLUMN_NAME])


def downgrade() -> None:
    """Downgrade schema."""
    bind = op.get_bind()
    inspector = inspect(bind)
    existing = {ix["name"] for ix in inspector.get_indexes(_TABLE_NAME)}
    if _INDEX_NAME in existing:
        op.drop_index(_INDEX_NAME, table_name=_TABLE_NAME)
