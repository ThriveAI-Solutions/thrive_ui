"""add organization column to thrive_user

Adds a single nullable VARCHAR(120) `organization` column to ``thrive_user``
so the bulk import and the admin create / edit / self-edit surfaces can
record which org each user belongs to. Email already exists on the User
model with `unique=True, collation="NOCASE"`; no schema change for email.

Existing rows survive the migration with ``organization = NULL`` (Epic #98
explicitly chose no backfill).

Revision ID: aa611613d67b
Revises: f95a95e2dbb1
Create Date: 2026-06-06 11:30:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

# revision identifiers, used by Alembic.
revision: str = "aa611613d67b"
down_revision: Union[str, Sequence[str], None] = "f95a95e2dbb1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    user_cols = {c["name"] for c in inspector.get_columns("thrive_user")}
    if "organization" not in user_cols:
        op.add_column(
            "thrive_user",
            sa.Column("organization", sa.String(length=120), nullable=True),
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    user_cols = {c["name"] for c in inspector.get_columns("thrive_user")}
    if "organization" in user_cols:
        op.drop_column("thrive_user", "organization")
