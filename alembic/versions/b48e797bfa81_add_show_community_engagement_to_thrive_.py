"""add show_community_engagement to thrive_user

Adds a single nullable BOOLEAN ``show_community_engagement`` column to
``thrive_user`` so the chat sidebar can gate the 📊 Community Engagement
popover behind a per-user preference (default OFF). Power users opt in
via the Settings dialog; existing users see the simpler sidebar by
default.

Existing rows survive the migration with ``show_community_engagement = NULL``
and are coerced to ``False`` by the defensive ``getattr(..., False) or False``
read in ``orm.functions.set_user_preferences_in_session_state``. No backfill
required.

Revision ID: b48e797bfa81
Revises: aa611613d67b
Create Date: 2026-06-10 15:52:32.843776

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

# revision identifiers, used by Alembic.
revision: str = "b48e797bfa81"
down_revision: Union[str, Sequence[str], None] = "aa611613d67b"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    user_cols = {c["name"] for c in inspector.get_columns("thrive_user")}
    if "show_community_engagement" not in user_cols:
        with op.batch_alter_table("thrive_user", schema=None) as batch_op:
            batch_op.add_column(
                sa.Column("show_community_engagement", sa.Boolean(), nullable=True)
            )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    user_cols = {c["name"] for c in inspector.get_columns("thrive_user")}
    if "show_community_engagement" in user_cols:
        with op.batch_alter_table("thrive_user", schema=None) as batch_op:
            batch_op.drop_column("show_community_engagement")
