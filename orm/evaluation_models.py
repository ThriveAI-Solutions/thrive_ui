"""Persistence models for agentic feedback and authenticated evaluations."""

from sqlalchemy import (
    TIMESTAMP,
    Boolean,
    CheckConstraint,
    Column,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)

from orm.models import Base


class AgentRunFeedback(Base):
    """Current owned feedback state for one completed agentic run."""

    __tablename__ = "thrive_agent_run_feedback"
    __table_args__ = (
        UniqueConstraint("agent_run_id", "user_id", name="uq_agent_run_feedback_owner"),
        CheckConstraint("rating IN ('up', 'down')", name="ck_agent_run_feedback_rating"),
        Index("ix_thrive_agent_run_feedback_agent_run", "agent_run_id"),
        Index("ix_thrive_agent_run_feedback_user", "user_id"),
        Index("ix_thrive_agent_run_feedback_rating", "rating"),
        Index("ix_thrive_agent_run_feedback_created", "created_at"),
    )

    id = Column(Integer, primary_key=True)
    agent_run_id = Column(Integer, ForeignKey("thrive_agent_run.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(Integer, ForeignKey("thrive_user.id", ondelete="CASCADE"), nullable=False)
    rating = Column(String(8), nullable=False)
    category = Column(String(64), nullable=True)
    comment = Column(String(500), nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now(), nullable=False)
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now(), nullable=False)


class AgentRunFeedbackEvent(Base):
    """Append-only audit event for feedback create/update/clear actions."""

    __tablename__ = "thrive_agent_run_feedback_event"
    __table_args__ = (
        Index("ix_thrive_agent_run_feedback_event_feedback", "feedback_id"),
        Index("ix_thrive_agent_run_feedback_event_actor", "actor_user_id"),
        Index("ix_thrive_agent_run_feedback_event_created", "created_at"),
    )

    id = Column(Integer, primary_key=True)
    feedback_id = Column(Integer, ForeignKey("thrive_agent_run_feedback.id", ondelete="CASCADE"), nullable=False)
    actor_user_id = Column(Integer, ForeignKey("thrive_user.id"), nullable=False)
    old_rating = Column(String(8), nullable=True)
    new_rating = Column(String(8), nullable=True)
    old_category = Column(String(64), nullable=True)
    new_category = Column(String(64), nullable=True)
    old_comment = Column(String(500), nullable=True)
    new_comment = Column(String(500), nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now(), nullable=False)


class EvaluationCase(Base):
    """Immutable versioned evaluation case snapshot."""

    __tablename__ = "thrive_evaluation_case"
    __table_args__ = (
        UniqueConstraint("case_id", "version", name="uq_evaluation_case_version"),
        CheckConstraint("source_type IN ('feedback', 'curated')", name="ck_evaluation_case_source_type"),
        Index("ix_thrive_evaluation_case_case_id", "case_id"),
        Index("ix_thrive_evaluation_case_source_status", "source_type", "status"),
        Index("ix_thrive_evaluation_case_source_feedback", "source_feedback_id"),
        Index("ix_thrive_evaluation_case_source_message", "source_message_id"),
        Index("ix_thrive_evaluation_case_source_agent_run", "source_agent_run_id"),
        Index("ix_thrive_evaluation_case_promoted_from", "promoted_from_case_id"),
        Index("ix_thrive_evaluation_case_created_by", "created_by"),
    )

    id = Column(Integer, primary_key=True)
    case_id = Column(String(36), nullable=False)
    version = Column(Integer, nullable=False)
    source_type = Column(String(16), nullable=False)
    status = Column(String(16), nullable=False)
    source_feedback_id = Column(Integer, ForeignKey("thrive_agent_run_feedback.id"), nullable=True)
    source_message_id = Column(Integer, ForeignKey("thrive_message.id"), nullable=True)
    source_agent_run_id = Column(Integer, ForeignKey("thrive_agent_run.id"), nullable=True)
    promoted_from_case_id = Column(Integer, ForeignKey("thrive_evaluation_case.id"), nullable=True)
    payload_json = Column(Text, nullable=False)
    created_by = Column(Integer, ForeignKey("thrive_user.id"), nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now(), nullable=False)
    expires_at = Column(TIMESTAMP, nullable=True)


class EvaluationRun(Base):
    """Durable launch and lifecycle state for evaluation execution."""

    __tablename__ = "thrive_evaluation_run"
    __table_args__ = (
        Index("ix_thrive_evaluation_run_run_id", "run_id", unique=True),
        Index("ix_thrive_evaluation_run_requested_by", "requested_by"),
        Index("ix_thrive_evaluation_run_status_time", "status", "created_at"),
        Index("ix_thrive_evaluation_run_heartbeat", "heartbeat_at"),
    )

    id = Column(Integer, primary_key=True)
    run_id = Column(String(36), nullable=False, unique=True)
    run_type = Column(String(24), nullable=False)
    execution_mode = Column(String(16), nullable=False)
    status = Column(String(32), nullable=False)
    requested_by = Column(Integer, ForeignKey("thrive_user.id"), nullable=False)
    total_cases = Column(Integer, nullable=False)
    completed_cases = Column(Integer, nullable=False, default=0)
    failed_cases = Column(Integer, nullable=False, default=0)
    cancel_requested = Column(Boolean, nullable=False, default=False)
    model_json = Column(Text, nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now(), nullable=False)
    started_at = Column(TIMESTAMP, nullable=True)
    heartbeat_at = Column(TIMESTAMP, nullable=True)
    completed_at = Column(TIMESTAMP, nullable=True)


class EvaluationCaseResult(Base):
    """Independent execution result for one case version in one evaluation run."""

    __tablename__ = "thrive_evaluation_case_result"
    __table_args__ = (
        UniqueConstraint("evaluation_run_id", "evaluation_case_id", "attempt", name="uq_eval_result_attempt"),
        Index("ix_thrive_evaluation_case_result_run_ordinal", "evaluation_run_id", "ordinal"),
        Index("ix_thrive_evaluation_case_result_case", "evaluation_case_id"),
        Index("ix_thrive_evaluation_case_result_status", "status"),
        Index("ix_thrive_evaluation_case_result_reviewer", "reviewed_by"),
    )

    id = Column(Integer, primary_key=True)
    evaluation_run_id = Column(Integer, ForeignKey("thrive_evaluation_run.id", ondelete="CASCADE"), nullable=False)
    evaluation_case_id = Column(Integer, ForeignKey("thrive_evaluation_case.id"), nullable=False)
    ordinal = Column(Integer, nullable=False)
    attempt = Column(Integer, nullable=False)
    status = Column(String(24), nullable=False)
    result_json = Column(Text, nullable=True)
    error_type = Column(String(100), nullable=True)
    error_message = Column(String(500), nullable=True)
    judge_verdict = Column(String(24), nullable=True)
    judge_reason = Column(String(500), nullable=True)
    judge_confidence = Column(Float, nullable=True)
    final_verdict = Column(String(24), nullable=True)
    review_note = Column(String(1000), nullable=True)
    reviewed_by = Column(Integer, ForeignKey("thrive_user.id"), nullable=True)
    reviewed_at = Column(TIMESTAMP, nullable=True)
    started_at = Column(TIMESTAMP, nullable=True)
    completed_at = Column(TIMESTAMP, nullable=True)


class EvaluationReviewEvent(Base):
    """Append-only admin verdict history for an evaluation result."""

    __tablename__ = "thrive_evaluation_review_event"
    __table_args__ = (
        CheckConstraint(
            "new_verdict IN ('correct', 'incorrect', 'cant_tell')",
            name="ck_evaluation_review_event_new_verdict",
        ),
        Index("ix_thrive_evaluation_review_event_result", "result_id"),
        Index("ix_thrive_evaluation_review_event_reviewer", "reviewer_id"),
        Index("ix_thrive_evaluation_review_event_created", "created_at"),
    )

    id = Column(Integer, primary_key=True)
    result_id = Column(Integer, ForeignKey("thrive_evaluation_case_result.id", ondelete="CASCADE"), nullable=False)
    reviewer_id = Column(Integer, ForeignKey("thrive_user.id"), nullable=False)
    old_verdict = Column(String(24), nullable=True)
    new_verdict = Column(String(24), nullable=False)
    note = Column(String(1000), nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now(), nullable=False)


class AdminNotification(Base):
    """PHI-free notification for Admin evaluation workspace events."""

    __tablename__ = "thrive_admin_notification"
    __table_args__ = (
        Index("ix_thrive_admin_notification_user", "user_id"),
        Index("ix_thrive_admin_notification_run", "evaluation_run_id"),
        Index("ix_thrive_admin_notification_unread", "user_id", "read_at", "created_at"),
    )

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("thrive_user.id", ondelete="CASCADE"), nullable=False)
    evaluation_run_id = Column(Integer, ForeignKey("thrive_evaluation_run.id", ondelete="CASCADE"), nullable=False)
    kind = Column(String(40), nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now(), nullable=False)
    read_at = Column(TIMESTAMP, nullable=True)
