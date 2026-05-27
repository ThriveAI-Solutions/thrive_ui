"""agentic run logging

Revision ID: f95a95e2dbb1
Revises: f235fb6fdd7f
Create Date: 2026-05-27 13:36:06.653614

"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


# revision identifiers, used by Alembic.
revision: str = "f95a95e2dbb1"
down_revision: Union[str, Sequence[str], None] = "f235fb6fdd7f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    existing_tables = set(inspector.get_table_names())
    tool_cols = {c["name"] for c in inspector.get_columns("thrive_tool_call")}

    # --- ToolCall enrichment columns (additive, all nullable / defaulted) ---
    new_tool_columns = [
        sa.Column("run_id", sa.String(length=36), nullable=True),
        sa.Column("tool_call_id", sa.String(length=100), nullable=True),
        sa.Column("call_index", sa.Integer(), nullable=True),
        sa.Column("turn_index", sa.Integer(), nullable=True),
        sa.Column("attempt_index", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("started_event_seq", sa.Integer(), nullable=True),
        sa.Column("completed_event_seq", sa.Integer(), nullable=True),
        sa.Column("result_json", sa.Text(), nullable=True),
        sa.Column("result_truncated", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("result_bytes", sa.Integer(), nullable=True),
        sa.Column("result_hash", sa.String(length=64), nullable=True),
        sa.Column("sql_executed_json", sa.Text(), nullable=True),
        sa.Column("sql_executed_truncated", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("sql_executed_bytes", sa.Integer(), nullable=True),
        sa.Column("sql_executed_hash", sa.String(length=64), nullable=True),
    ]
    for col in new_tool_columns:
        if col.name not in tool_cols:
            op.add_column("thrive_tool_call", col)
    if "run_id" not in tool_cols:
        op.create_index("ix_thrive_tool_call_run_id", "thrive_tool_call", ["run_id"])
        op.create_index("ix_thrive_tool_call_run_call", "thrive_tool_call", ["run_id", "call_index"])
        op.create_index("ix_thrive_tool_call_tool_call_id", "thrive_tool_call", ["tool_call_id"])

    if "thrive_agent_run" not in existing_tables:
        op.create_table(
            "thrive_agent_run",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("run_id", sa.String(length=36), nullable=False),
            sa.Column("session_id", sa.String(length=64), nullable=False),
            sa.Column("group_id", sa.String(length=50), nullable=True),
            sa.Column("parent_run_id", sa.String(length=36), nullable=True),
            sa.Column("resume_reason", sa.String(length=40), nullable=True),
            sa.Column("user_message_id", sa.Integer(), nullable=True),
            sa.Column("final_message_id", sa.Integer(), nullable=True),
            sa.Column("user_id", sa.Integer(), nullable=False),
            sa.Column("user_role", sa.Integer(), nullable=False),
            sa.Column("question", sa.Text(), nullable=True),
            sa.Column("selected_patient_source_id", sa.String(length=50), nullable=True),
            sa.Column("selected_patient_display_name", sa.String(length=255), nullable=True),
            sa.Column("selected_patient_dob", sa.String(length=50), nullable=True),
            sa.Column("selected_patient_selection_origin", sa.String(length=32), nullable=True),
            sa.Column("llm_provider", sa.String(length=50), nullable=True),
            sa.Column("llm_model", sa.String(length=100), nullable=True),
            sa.Column("model_settings_json", sa.Text(), nullable=True),
            sa.Column("system_prompt_hash", sa.String(length=64), nullable=True),
            sa.Column("tool_schema_hash", sa.String(length=64), nullable=True),
            sa.Column("message_history_json", sa.Text(), nullable=True),
            sa.Column("final_answer_text", sa.Text(), nullable=True),
            sa.Column("input_tokens", sa.Integer(), nullable=True),
            sa.Column("output_tokens", sa.Integer(), nullable=True),
            sa.Column("total_tokens", sa.Integer(), nullable=True),
            sa.Column("tool_call_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("event_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("total_elapsed_ms", sa.Integer(), nullable=True),
            sa.Column("cap_reached", sa.String(length=20), nullable=True),
            sa.Column("status", sa.String(length=20), nullable=False, server_default="open"),
            sa.Column("success", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("error_type", sa.String(length=100), nullable=True),
            sa.Column("error", sa.Text(), nullable=True),
            sa.Column("stack_trace", sa.Text(), nullable=True),
            sa.Column("review_status", sa.String(length=20), nullable=False, server_default="unreviewed"),
            sa.Column("reviewed_by", sa.Integer(), nullable=True),
            sa.Column("reviewed_at", sa.TIMESTAMP(), nullable=True),
            sa.Column("review_notes", sa.Text(), nullable=True),
            sa.Column("issue_url", sa.Text(), nullable=True),
            sa.Column("logging_mode", sa.String(length=20), nullable=False, server_default="full"),
            sa.Column("schema_version", sa.Integer(), nullable=False, server_default="1"),
            sa.Column("app_git_sha", sa.String(length=40), nullable=True),
            sa.Column("environment", sa.String(length=50), nullable=True),
            sa.Column("created_at", sa.TIMESTAMP(), server_default=sa.func.now(), nullable=False),
            sa.Column("completed_at", sa.TIMESTAMP(), nullable=True),
            sa.ForeignKeyConstraint(["user_id"], ["thrive_user.id"]),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_thrive_agent_run_run_id", "thrive_agent_run", ["run_id"], unique=True)
        op.create_index("ix_thrive_agent_run_session_id", "thrive_agent_run", ["session_id"])
        op.create_index("ix_thrive_agent_run_user_id", "thrive_agent_run", ["user_id"])
        op.create_index("ix_thrive_agent_run_created_at", "thrive_agent_run", ["created_at"])
        op.create_index("ix_thrive_agent_run_group_id", "thrive_agent_run", ["group_id"])
        op.create_index("ix_thrive_agent_run_selected_patient", "thrive_agent_run", ["selected_patient_source_id"])
        op.create_index("ix_thrive_agent_run_status", "thrive_agent_run", ["status"])
        op.create_index("ix_thrive_agent_run_review_status", "thrive_agent_run", ["review_status"])

    if "thrive_agent_run_event" not in existing_tables:
        op.create_table(
            "thrive_agent_run_event",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("run_id", sa.String(length=36), nullable=False),
            sa.Column("seq", sa.Integer(), nullable=False),
            sa.Column("event_type", sa.String(length=50), nullable=False),
            sa.Column("turn_index", sa.Integer(), nullable=True),
            sa.Column("tool_call_id", sa.String(length=100), nullable=True),
            sa.Column("tool_name", sa.String(length=64), nullable=True),
            sa.Column("payload_json", sa.Text(), nullable=True),
            sa.Column("payload_summary", sa.Text(), nullable=True),
            sa.Column("payload_truncated", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("payload_bytes", sa.Integer(), nullable=True),
            sa.Column("payload_hash", sa.String(length=64), nullable=True),
            sa.Column("elapsed_ms", sa.Integer(), nullable=True),
            sa.Column("created_at", sa.TIMESTAMP(), server_default=sa.func.now(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_thrive_agent_run_event_run_seq", "thrive_agent_run_event", ["run_id", "seq"], unique=True)
        op.create_index("ix_thrive_agent_run_event_run_id", "thrive_agent_run_event", ["run_id"])
        op.create_index("ix_thrive_agent_run_event_type", "thrive_agent_run_event", ["event_type"])
        op.create_index("ix_thrive_agent_run_event_tool_name", "thrive_agent_run_event", ["tool_name"])
        op.create_index("ix_thrive_agent_run_event_created_at", "thrive_agent_run_event", ["created_at"])

    if "thrive_agent_patient_access" not in existing_tables:
        op.create_table(
            "thrive_agent_patient_access",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("run_id", sa.String(length=36), nullable=True),
            sa.Column("tool_call_id", sa.String(length=100), nullable=True),
            sa.Column("event_seq", sa.Integer(), nullable=True),
            sa.Column("session_id", sa.String(length=64), nullable=False),
            sa.Column("user_id", sa.Integer(), nullable=False),
            sa.Column("source_id", sa.String(length=50), nullable=False),
            sa.Column("display_name", sa.String(length=255), nullable=True),
            sa.Column("access_type", sa.String(length=40), nullable=False),
            sa.Column("access_origin", sa.String(length=40), nullable=False),
            sa.Column("tool_name", sa.String(length=64), nullable=True),
            sa.Column("created_at", sa.TIMESTAMP(), server_default=sa.func.now(), nullable=False),
            sa.ForeignKeyConstraint(["user_id"], ["thrive_user.id"]),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_thrive_agent_patient_access_source", "thrive_agent_patient_access", ["source_id"])
        op.create_index("ix_thrive_agent_patient_access_user", "thrive_agent_patient_access", ["user_id"])
        op.create_index("ix_thrive_agent_patient_access_run", "thrive_agent_patient_access", ["run_id"])
        op.create_index("ix_thrive_agent_patient_access_tool_call", "thrive_agent_patient_access", ["tool_call_id"])
        op.create_index("ix_thrive_agent_patient_access_created", "thrive_agent_patient_access", ["created_at"])
        op.create_index("ix_thrive_agent_patient_access_type", "thrive_agent_patient_access", ["access_type"])

    if "thrive_patient_selection_event" not in existing_tables:
        op.create_table(
            "thrive_patient_selection_event",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("session_id", sa.String(length=64), nullable=False),
            sa.Column("run_id", sa.String(length=36), nullable=True),
            sa.Column("user_id", sa.Integer(), nullable=False),
            sa.Column("source_id", sa.String(length=50), nullable=True),
            sa.Column("previous_source_id", sa.String(length=50), nullable=True),
            sa.Column("display_name", sa.String(length=255), nullable=True),
            sa.Column("selection_origin", sa.String(length=32), nullable=False),
            sa.Column("action", sa.String(length=20), nullable=False),
            sa.Column("created_at", sa.TIMESTAMP(), server_default=sa.func.now(), nullable=False),
            sa.ForeignKeyConstraint(["user_id"], ["thrive_user.id"]),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_thrive_patient_selection_session", "thrive_patient_selection_event", ["session_id"])
        op.create_index("ix_thrive_patient_selection_user", "thrive_patient_selection_event", ["user_id"])
        op.create_index("ix_thrive_patient_selection_source", "thrive_patient_selection_event", ["source_id"])
        op.create_index("ix_thrive_patient_selection_created", "thrive_patient_selection_event", ["created_at"])


def downgrade() -> None:
    op.drop_table("thrive_patient_selection_event")
    op.drop_table("thrive_agent_patient_access")
    op.drop_table("thrive_agent_run_event")
    op.drop_table("thrive_agent_run")
    with op.batch_alter_table("thrive_tool_call") as batch:
        for ix in ("ix_thrive_tool_call_tool_call_id", "ix_thrive_tool_call_run_call", "ix_thrive_tool_call_run_id"):
            batch.drop_index(ix)
        for col in (
            "sql_executed_hash",
            "sql_executed_bytes",
            "sql_executed_truncated",
            "sql_executed_json",
            "result_hash",
            "result_bytes",
            "result_truncated",
            "result_json",
            "completed_event_seq",
            "started_event_seq",
            "attempt_index",
            "turn_index",
            "call_index",
            "tool_call_id",
            "run_id",
        ):
            batch.drop_column(col)
