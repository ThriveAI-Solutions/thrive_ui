"""agentic_foundations

Adds:
- thrive_user.agentic_mode column (Boolean, default False)
- thrive_tool_call table (audit trail per spec §8.6)
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# KEEP the auto-generated revision hash from the file header
revision: str = "f3e688a55df6"
down_revision: Union[str, Sequence[str], None] = "2540d625d0fe"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "thrive_user",
        sa.Column("agentic_mode", sa.Boolean(), nullable=True, server_default=sa.false()),
    )
    op.create_table(
        "thrive_tool_call",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("user_role", sa.Integer(), nullable=False),
        sa.Column("selected_patient_source_id", sa.String(length=50), nullable=True),
        sa.Column("tool_name", sa.String(length=64), nullable=False),
        sa.Column("arguments_json", sa.Text(), nullable=False),
        sa.Column("result_summary", sa.Text(), nullable=False),
        sa.Column("elapsed_ms", sa.Integer(), nullable=False),
        sa.Column("success", sa.Boolean(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.TIMESTAMP(), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["thrive_user.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_thrive_tool_call_session_id", "thrive_tool_call", ["session_id"])
    op.create_index("ix_thrive_tool_call_user_id", "thrive_tool_call", ["user_id"])
    op.create_index("ix_thrive_tool_call_created_at", "thrive_tool_call", ["created_at"])
    op.create_index("ix_thrive_tool_call_tool_name", "thrive_tool_call", ["tool_name"])
    op.create_index("ix_thrive_tool_call_selected_patient", "thrive_tool_call", ["selected_patient_source_id"])


def downgrade() -> None:
    op.drop_index("ix_thrive_tool_call_selected_patient", table_name="thrive_tool_call")
    op.drop_index("ix_thrive_tool_call_tool_name", table_name="thrive_tool_call")
    op.drop_index("ix_thrive_tool_call_created_at", table_name="thrive_tool_call")
    op.drop_index("ix_thrive_tool_call_user_id", table_name="thrive_tool_call")
    op.drop_index("ix_thrive_tool_call_session_id", table_name="thrive_tool_call")
    op.drop_table("thrive_tool_call")
    op.drop_column("thrive_user", "agentic_mode")
