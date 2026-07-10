"""Owner-authorized persistence helpers for agentic run thumbs feedback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from sqlalchemy.orm import Session

from orm.evaluation_models import AgentRunFeedback, AgentRunFeedbackEvent
from orm.models import AgentRun

FeedbackRating = Literal["up", "down"]

AGENT_FEEDBACK_CATEGORIES = (
    "Incorrect answer",
    "Wrong patient or cohort",
    "Missing or incomplete data",
    "Misleading explanation",
    "Unsafe or inappropriate response",
    "Other",
)

COMMENT_MAX_CHARS = 500
_COMPLETED_STATUSES = frozenset({"success"})


@dataclass(frozen=True)
class AgentFeedbackState:
    """Current feedback state shown back to the owner."""

    run_id: str
    rating: FeedbackRating
    category: str | None = None
    comment: str | None = None


class AgentFeedbackError(ValueError):
    """Raised for invalid or unauthorized feedback mutations."""


def _normalize_comment(comment: str | None) -> str | None:
    if comment is None:
        return None
    normalized = str(comment).strip()
    if not normalized:
        return None
    if len(normalized) > COMMENT_MAX_CHARS:
        raise AgentFeedbackError(f"Feedback comment must be {COMMENT_MAX_CHARS} characters or fewer.")
    return normalized


def _normalize_category(rating: str, category: str | None) -> str | None:
    if rating == "up":
        return None
    normalized = (category or "").strip()
    if normalized not in AGENT_FEEDBACK_CATEGORIES:
        raise AgentFeedbackError("Thumbs-down feedback requires an approved category.")
    return normalized


def _owned_completed_run(session: Session, *, run_id: str, user_id: int) -> AgentRun:
    run = session.query(AgentRun).filter(AgentRun.run_id == run_id).first()
    if run is None:
        raise AgentFeedbackError("Agent run not found.")
    if int(run.user_id) != int(user_id):
        raise AgentFeedbackError("Only the agent run owner can change feedback.")
    if run.status not in _COMPLETED_STATUSES or not run.final_message_id:
        raise AgentFeedbackError("Feedback is only available for completed agent answers.")
    return run


def get_agent_run_feedback(session: Session, *, run_id: str, user_id: int) -> AgentFeedbackState | None:
    """Return the owner's current feedback for a run, if any.

    Cross-user reads intentionally return ``None`` rather than leaking whether
    another user left feedback.
    """
    run = session.query(AgentRun).filter(AgentRun.run_id == run_id, AgentRun.user_id == user_id).first()
    if run is None:
        return None
    row = (
        session.query(AgentRunFeedback)
        .filter(AgentRunFeedback.agent_run_id == run.id, AgentRunFeedback.user_id == user_id)
        .first()
    )
    if row is None:
        return None
    return AgentFeedbackState(run_id=run.run_id, rating=row.rating, category=row.category, comment=row.comment)


def set_agent_run_feedback(
    session: Session,
    *,
    run_id: str,
    user_id: int,
    rating: FeedbackRating,
    category: str | None = None,
    comment: str | None = None,
) -> AgentFeedbackState:
    """Create or update owner feedback transactionally and append an audit event."""
    if rating not in ("up", "down"):
        raise AgentFeedbackError("Feedback rating must be 'up' or 'down'.")
    normalized_category = _normalize_category(rating, category)
    normalized_comment = _normalize_comment(comment)
    if rating == "up":
        normalized_comment = None

    try:
        run = _owned_completed_run(session, run_id=run_id, user_id=user_id)
        row = (
            session.query(AgentRunFeedback)
            .filter(AgentRunFeedback.agent_run_id == run.id, AgentRunFeedback.user_id == user_id)
            .with_for_update()
            .first()
        )
        if row is None:
            row = AgentRunFeedback(agent_run_id=run.id, user_id=user_id, rating=rating)
            session.add(row)
            session.flush()
            old_rating = old_category = old_comment = None
        else:
            old_rating = row.rating
            old_category = row.category
            old_comment = row.comment

        row.rating = rating
        row.category = normalized_category
        row.comment = normalized_comment
        session.add(row)
        session.add(
            AgentRunFeedbackEvent(
                feedback_id=row.id,
                actor_user_id=user_id,
                old_rating=old_rating,
                new_rating=rating,
                old_category=old_category,
                new_category=normalized_category,
                old_comment=old_comment,
                new_comment=normalized_comment,
            )
        )
        session.commit()
        return AgentFeedbackState(run_id=run.run_id, rating=row.rating, category=row.category, comment=row.comment)
    except Exception:
        session.rollback()
        raise


def clear_agent_run_feedback(session: Session, *, run_id: str, user_id: int) -> None:
    """Clear the owner's current feedback and append a clear audit event."""
    try:
        run = _owned_completed_run(session, run_id=run_id, user_id=user_id)
        row = (
            session.query(AgentRunFeedback)
            .filter(AgentRunFeedback.agent_run_id == run.id, AgentRunFeedback.user_id == user_id)
            .with_for_update()
            .first()
        )
        if row is None:
            session.commit()
            return
        session.add(
            AgentRunFeedbackEvent(
                feedback_id=row.id,
                actor_user_id=user_id,
                old_rating=row.rating,
                new_rating=None,
                old_category=row.category,
                new_category=None,
                old_comment=row.comment,
                new_comment=None,
            )
        )
        session.delete(row)
        session.commit()
    except Exception:
        session.rollback()
        raise


def completed_owned_run_for_group(
    session: Session,
    *,
    group_id: str,
    user_id: int,
    message_ids: set[int],
) -> AgentRun | None:
    """Find the latest completed owned AgentRun whose final message is in a rendered group."""
    if not group_id or not message_ids:
        return None
    return (
        session.query(AgentRun)
        .filter(
            AgentRun.group_id == group_id,
            AgentRun.user_id == user_id,
            AgentRun.status.in_(_COMPLETED_STATUSES),
            AgentRun.final_message_id.in_(message_ids),
        )
        .order_by(AgentRun.completed_at.desc(), AgentRun.id.desc())
        .first()
    )
