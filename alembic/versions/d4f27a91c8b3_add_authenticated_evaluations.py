"""add authenticated evaluation workspace tables

Revision ID: d4f27a91c8b3
Revises: c233fb500001
Create Date: 2026-07-10

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "d4f27a91c8b3"
down_revision: Union[str, Sequence[str], None] = "c233fb500001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create every table backing agentic feedback and authenticated evaluations
    (orm/evaluation_models.py). Objects are created in dependency order: feedback
    tables first (only depend on pre-existing thrive_agent_run/thrive_user),
    then evaluation_case (also depends on thrive_message and itself), then
    evaluation_run, then the tables that depend on both evaluation_run and
    evaluation_case, then the leaf tables.
    """
    op.create_table(
        "thrive_agent_run_feedback",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("agent_run_id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("rating", sa.String(length=8), nullable=False),
        sa.Column("category", sa.String(length=64), nullable=True),
        sa.Column("comment", sa.String(length=500), nullable=True),
        sa.Column("created_at", sa.TIMESTAMP(), server_default=sa.func.now(), nullable=False),
        sa.Column(
            "updated_at", sa.TIMESTAMP(), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(["agent_run_id"], ["thrive_agent_run.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["thrive_user.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("agent_run_id", "user_id", name="uq_agent_run_feedback_owner"),
        sa.CheckConstraint("rating IN ('up', 'down')", name="ck_agent_run_feedback_rating"),
    )
    op.create_index(
        "ix_thrive_agent_run_feedback_agent_run", "thrive_agent_run_feedback", ["agent_run_id"]
    )
    op.create_index("ix_thrive_agent_run_feedback_user", "thrive_agent_run_feedback", ["user_id"])
    op.create_index("ix_thrive_agent_run_feedback_rating", "thrive_agent_run_feedback", ["rating"])
    op.create_index("ix_thrive_agent_run_feedback_created", "thrive_agent_run_feedback", ["created_at"])

    op.create_table(
        "thrive_agent_run_feedback_event",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("feedback_id", sa.Integer(), nullable=False),
        sa.Column("actor_user_id", sa.Integer(), nullable=False),
        sa.Column("old_rating", sa.String(length=8), nullable=True),
        sa.Column("new_rating", sa.String(length=8), nullable=True),
        sa.Column("old_category", sa.String(length=64), nullable=True),
        sa.Column("new_category", sa.String(length=64), nullable=True),
        sa.Column("old_comment", sa.String(length=500), nullable=True),
        sa.Column("new_comment", sa.String(length=500), nullable=True),
        sa.Column("created_at", sa.TIMESTAMP(), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["feedback_id"], ["thrive_agent_run_feedback.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["actor_user_id"], ["thrive_user.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_thrive_agent_run_feedback_event_feedback", "thrive_agent_run_feedback_event", ["feedback_id"]
    )
    op.create_index(
        "ix_thrive_agent_run_feedback_event_actor", "thrive_agent_run_feedback_event", ["actor_user_id"]
    )
    op.create_index(
        "ix_thrive_agent_run_feedback_event_created", "thrive_agent_run_feedback_event", ["created_at"]
    )

    op.create_table(
        "thrive_evaluation_case",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("case_id", sa.String(length=36), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("source_type", sa.String(length=16), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("source_feedback_id", sa.Integer(), nullable=True),
        sa.Column("source_message_id", sa.Integer(), nullable=True),
        sa.Column("source_agent_run_id", sa.Integer(), nullable=True),
        sa.Column("promoted_from_case_id", sa.Integer(), nullable=True),
        sa.Column("payload_json", sa.Text(), nullable=False),
        sa.Column("created_by", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(), server_default=sa.func.now(), nullable=False),
        sa.Column("expires_at", sa.TIMESTAMP(), nullable=True),
        sa.ForeignKeyConstraint(["source_feedback_id"], ["thrive_agent_run_feedback.id"]),
        sa.ForeignKeyConstraint(["source_message_id"], ["thrive_message.id"]),
        sa.ForeignKeyConstraint(["source_agent_run_id"], ["thrive_agent_run.id"]),
        sa.ForeignKeyConstraint(["promoted_from_case_id"], ["thrive_evaluation_case.id"]),
        sa.ForeignKeyConstraint(["created_by"], ["thrive_user.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("case_id", "version", name="uq_evaluation_case_version"),
        sa.CheckConstraint("source_type IN ('feedback', 'curated')", name="ck_evaluation_case_source_type"),
    )
    op.create_index("ix_thrive_evaluation_case_case_id", "thrive_evaluation_case", ["case_id"])
    op.create_index(
        "ix_thrive_evaluation_case_source_status", "thrive_evaluation_case", ["source_type", "status"]
    )
    op.create_index(
        "ix_thrive_evaluation_case_source_feedback", "thrive_evaluation_case", ["source_feedback_id"]
    )
    op.create_index(
        "ix_thrive_evaluation_case_source_message", "thrive_evaluation_case", ["source_message_id"]
    )
    op.create_index(
        "ix_thrive_evaluation_case_source_agent_run", "thrive_evaluation_case", ["source_agent_run_id"]
    )
    op.create_index(
        "ix_thrive_evaluation_case_promoted_from", "thrive_evaluation_case", ["promoted_from_case_id"]
    )
    op.create_index("ix_thrive_evaluation_case_created_by", "thrive_evaluation_case", ["created_by"])

    op.create_table(
        "thrive_evaluation_run",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.String(length=36), nullable=False, unique=True),
        sa.Column("run_type", sa.String(length=24), nullable=False),
        sa.Column("execution_mode", sa.String(length=16), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("requested_by", sa.Integer(), nullable=False),
        sa.Column("total_cases", sa.Integer(), nullable=False),
        sa.Column("completed_cases", sa.Integer(), nullable=False),
        sa.Column("failed_cases", sa.Integer(), nullable=False),
        sa.Column("cancel_requested", sa.Boolean(), nullable=False),
        sa.Column("model_json", sa.Text(), nullable=True),
        sa.Column("created_at", sa.TIMESTAMP(), server_default=sa.func.now(), nullable=False),
        sa.Column("started_at", sa.TIMESTAMP(), nullable=True),
        sa.Column("heartbeat_at", sa.TIMESTAMP(), nullable=True),
        sa.Column("completed_at", sa.TIMESTAMP(), nullable=True),
        sa.ForeignKeyConstraint(["requested_by"], ["thrive_user.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_thrive_evaluation_run_run_id", "thrive_evaluation_run", ["run_id"], unique=True
    )
    op.create_index("ix_thrive_evaluation_run_requested_by", "thrive_evaluation_run", ["requested_by"])
    op.create_index(
        "ix_thrive_evaluation_run_status_time", "thrive_evaluation_run", ["status", "created_at"]
    )
    op.create_index("ix_thrive_evaluation_run_heartbeat", "thrive_evaluation_run", ["heartbeat_at"])

    op.create_table(
        "thrive_evaluation_case_result",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("evaluation_run_id", sa.Integer(), nullable=False),
        sa.Column("evaluation_case_id", sa.Integer(), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("result_json", sa.Text(), nullable=True),
        sa.Column("error_type", sa.String(length=100), nullable=True),
        sa.Column("error_message", sa.String(length=500), nullable=True),
        sa.Column("judge_verdict", sa.String(length=24), nullable=True),
        sa.Column("judge_reason", sa.String(length=500), nullable=True),
        sa.Column("judge_confidence", sa.Float(), nullable=True),
        sa.Column("final_verdict", sa.String(length=24), nullable=True),
        sa.Column("review_note", sa.String(length=1000), nullable=True),
        sa.Column("reviewed_by", sa.Integer(), nullable=True),
        sa.Column("reviewed_at", sa.TIMESTAMP(), nullable=True),
        sa.Column("started_at", sa.TIMESTAMP(), nullable=True),
        sa.Column("completed_at", sa.TIMESTAMP(), nullable=True),
        sa.ForeignKeyConstraint(["evaluation_run_id"], ["thrive_evaluation_run.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["evaluation_case_id"], ["thrive_evaluation_case.id"]),
        sa.ForeignKeyConstraint(["reviewed_by"], ["thrive_user.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "evaluation_run_id", "evaluation_case_id", "attempt", name="uq_eval_result_attempt"
        ),
    )
    op.create_index(
        "ix_thrive_evaluation_case_result_run_ordinal",
        "thrive_evaluation_case_result",
        ["evaluation_run_id", "ordinal"],
    )
    op.create_index(
        "ix_thrive_evaluation_case_result_case", "thrive_evaluation_case_result", ["evaluation_case_id"]
    )
    op.create_index(
        "ix_thrive_evaluation_case_result_status", "thrive_evaluation_case_result", ["status"]
    )
    op.create_index(
        "ix_thrive_evaluation_case_result_reviewer", "thrive_evaluation_case_result", ["reviewed_by"]
    )

    op.create_table(
        "thrive_evaluation_review_event",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("result_id", sa.Integer(), nullable=False),
        sa.Column("reviewer_id", sa.Integer(), nullable=False),
        sa.Column("old_verdict", sa.String(length=24), nullable=True),
        sa.Column("new_verdict", sa.String(length=24), nullable=False),
        sa.Column("note", sa.String(length=1000), nullable=True),
        sa.Column("created_at", sa.TIMESTAMP(), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["result_id"], ["thrive_evaluation_case_result.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["reviewer_id"], ["thrive_user.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(
            "new_verdict IN ('correct', 'incorrect', 'cant_tell')",
            name="ck_evaluation_review_event_new_verdict",
        ),
    )
    op.create_index(
        "ix_thrive_evaluation_review_event_result", "thrive_evaluation_review_event", ["result_id"]
    )
    op.create_index(
        "ix_thrive_evaluation_review_event_reviewer", "thrive_evaluation_review_event", ["reviewer_id"]
    )
    op.create_index(
        "ix_thrive_evaluation_review_event_created", "thrive_evaluation_review_event", ["created_at"]
    )

    op.create_table(
        "thrive_admin_notification",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("evaluation_run_id", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(length=40), nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(), server_default=sa.func.now(), nullable=False),
        sa.Column("read_at", sa.TIMESTAMP(), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["thrive_user.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["evaluation_run_id"], ["thrive_evaluation_run.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_thrive_admin_notification_user", "thrive_admin_notification", ["user_id"])
    op.create_index(
        "ix_thrive_admin_notification_run", "thrive_admin_notification", ["evaluation_run_id"]
    )
    op.create_index(
        "ix_thrive_admin_notification_unread",
        "thrive_admin_notification",
        ["user_id", "read_at", "created_at"],
    )


def downgrade() -> None:
    """Drop every object created by upgrade(), in reverse dependency order."""
    op.drop_index("ix_thrive_admin_notification_unread", table_name="thrive_admin_notification")
    op.drop_index("ix_thrive_admin_notification_run", table_name="thrive_admin_notification")
    op.drop_index("ix_thrive_admin_notification_user", table_name="thrive_admin_notification")
    op.drop_table("thrive_admin_notification")

    op.drop_index(
        "ix_thrive_evaluation_review_event_created", table_name="thrive_evaluation_review_event"
    )
    op.drop_index(
        "ix_thrive_evaluation_review_event_reviewer", table_name="thrive_evaluation_review_event"
    )
    op.drop_index(
        "ix_thrive_evaluation_review_event_result", table_name="thrive_evaluation_review_event"
    )
    op.drop_table("thrive_evaluation_review_event")

    op.drop_index(
        "ix_thrive_evaluation_case_result_reviewer", table_name="thrive_evaluation_case_result"
    )
    op.drop_index(
        "ix_thrive_evaluation_case_result_status", table_name="thrive_evaluation_case_result"
    )
    op.drop_index("ix_thrive_evaluation_case_result_case", table_name="thrive_evaluation_case_result")
    op.drop_index(
        "ix_thrive_evaluation_case_result_run_ordinal", table_name="thrive_evaluation_case_result"
    )
    op.drop_table("thrive_evaluation_case_result")

    op.drop_index("ix_thrive_evaluation_run_heartbeat", table_name="thrive_evaluation_run")
    op.drop_index("ix_thrive_evaluation_run_status_time", table_name="thrive_evaluation_run")
    op.drop_index("ix_thrive_evaluation_run_requested_by", table_name="thrive_evaluation_run")
    op.drop_index("ix_thrive_evaluation_run_run_id", table_name="thrive_evaluation_run")
    op.drop_table("thrive_evaluation_run")

    op.drop_index("ix_thrive_evaluation_case_created_by", table_name="thrive_evaluation_case")
    op.drop_index("ix_thrive_evaluation_case_promoted_from", table_name="thrive_evaluation_case")
    op.drop_index("ix_thrive_evaluation_case_source_agent_run", table_name="thrive_evaluation_case")
    op.drop_index("ix_thrive_evaluation_case_source_message", table_name="thrive_evaluation_case")
    op.drop_index("ix_thrive_evaluation_case_source_feedback", table_name="thrive_evaluation_case")
    op.drop_index("ix_thrive_evaluation_case_source_status", table_name="thrive_evaluation_case")
    op.drop_index("ix_thrive_evaluation_case_case_id", table_name="thrive_evaluation_case")
    op.drop_table("thrive_evaluation_case")

    op.drop_index(
        "ix_thrive_agent_run_feedback_event_created", table_name="thrive_agent_run_feedback_event"
    )
    op.drop_index(
        "ix_thrive_agent_run_feedback_event_actor", table_name="thrive_agent_run_feedback_event"
    )
    op.drop_index(
        "ix_thrive_agent_run_feedback_event_feedback", table_name="thrive_agent_run_feedback_event"
    )
    op.drop_table("thrive_agent_run_feedback_event")

    op.drop_index("ix_thrive_agent_run_feedback_created", table_name="thrive_agent_run_feedback")
    op.drop_index("ix_thrive_agent_run_feedback_rating", table_name="thrive_agent_run_feedback")
    op.drop_index("ix_thrive_agent_run_feedback_user", table_name="thrive_agent_run_feedback")
    op.drop_index("ix_thrive_agent_run_feedback_agent_run", table_name="thrive_agent_run_feedback")
    op.drop_table("thrive_agent_run_feedback")
