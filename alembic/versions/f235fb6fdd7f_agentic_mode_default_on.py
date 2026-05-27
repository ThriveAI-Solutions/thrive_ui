"""agentic_mode_default_on

Revision ID: f235fb6fdd7f
Revises: f3e688a55df6
Create Date: 2026-05-27 11:53:41.398569

"""

from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "f235fb6fdd7f"
down_revision: Union[str, Sequence[str], None] = "f3e688a55df6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Turn agentic mode on for everybody.

    Data-only migration: flips every existing user (including any with a NULL
    value) to agentic_mode = True. New users get True via the ORM column
    default in orm/models.py. The column's server_default is intentionally
    left as-is since all inserts go through the ORM.
    """
    op.execute("UPDATE thrive_user SET agentic_mode = 1 WHERE agentic_mode IS NULL OR agentic_mode = 0")


def downgrade() -> None:
    """No-op: a data migration that turned a preference on is not safely
    reversible — we can't tell which users were on before. Downgrading the
    schema is handled by the previous revision."""
    pass
