"""Streamlit controls for owner-authorized agentic run feedback."""

from __future__ import annotations

import json

from typing import Literal

import streamlit as st

from orm.agent_feedback import (
    AGENT_FEEDBACK_CATEGORIES,
    AgentFeedbackError,
    clear_agent_run_feedback,
    completed_owned_run_for_group,
    get_agent_run_feedback,
    set_agent_run_feedback,
)
from orm.models import SessionLocal

_CATEGORY_PLACEHOLDER = "Select a category..."


def _user_id_from_session_state() -> int | None:
    direct = st.session_state.get("user_id")
    if direct is not None:
        try:
            return int(direct)
        except (TypeError, ValueError):
            return None
    cookies = st.session_state.get("cookies")
    if cookies is None:
        return None
    raw = cookies.get("user_id")
    if raw is None:
        return None
    try:
        return int(json.loads(raw))
    except (TypeError, ValueError, json.JSONDecodeError):
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None


def _message_ids(messages: list) -> set[int]:
    ids: set[int] = set()
    for msg in messages or []:
        msg_id = getattr(msg, "id", None)
        if msg_id is not None:
            try:
                ids.add(int(msg_id))
            except (TypeError, ValueError):
                pass
    return ids


def render_agent_feedback_for_group(messages: list) -> None:
    """Render exactly one thumbs control set for a completed agentic answer group.

    The controls are anchored after the grouped answer rather than on individual
    intermediate cards. A group is eligible only when an owned successful
    ``AgentRun`` points at a final message in that same rendered group.
    """
    if not messages:
        return
    group_id = getattr(messages[0], "group_id", None)
    if not group_id:
        return
    user_id = _user_id_from_session_state()
    if user_id is None:
        return

    session = SessionLocal()
    try:
        run = completed_owned_run_for_group(
            session,
            group_id=group_id,
            user_id=user_id,
            message_ids=_message_ids(messages),
        )
        if run is None:
            return
        feedback = get_agent_run_feedback(session, run_id=run.run_id, user_id=user_id)
    finally:
        session.close()

    cols = st.columns([0.1, 0.1, 0.1, 0.7])
    with cols[0]:
        if st.button(
            "👍",
            key=f"agent_feedback_up_{run.run_id}",
            type="primary" if feedback and feedback.rating == "up" else "secondary",
            help="Mark this agent answer as helpful",
        ):
            _submit_feedback(run.run_id, user_id, rating="up")
    with cols[1]:
        _render_down_popover(run.run_id, user_id, feedback)
    with cols[2]:
        if feedback is not None and st.button("Clear", key=f"agent_feedback_clear_{run.run_id}"):
            _clear_feedback(run.run_id, user_id)


def _submit_feedback(
    run_id: str,
    user_id: int,
    *,
    rating: Literal["up", "down"],
    category: str | None = None,
    comment: str | None = None,
) -> None:
    session = SessionLocal()
    try:
        set_agent_run_feedback(
            session,
            run_id=run_id,
            user_id=user_id,
            rating=rating,
            category=category,
            comment=comment,
        )
        st.toast("Feedback saved.")
        st.rerun()
    except AgentFeedbackError as exc:
        st.error(str(exc))
    finally:
        session.close()


def _clear_feedback(run_id: str, user_id: int) -> None:
    session = SessionLocal()
    try:
        clear_agent_run_feedback(session, run_id=run_id, user_id=user_id)
        st.toast("Feedback cleared.")
        st.rerun()
    except AgentFeedbackError as exc:
        st.error(str(exc))
    finally:
        session.close()


def _render_down_popover(run_id: str, user_id: int, feedback) -> None:
    icon = "👎✓" if feedback and feedback.rating == "down" else "👎"
    with st.popover(icon, width="content"):
        if feedback and feedback.rating == "down":
            detail = feedback.category or "Thumbs down"
            if feedback.comment:
                detail = f"{detail}: {feedback.comment}"
            st.info(f"Previous feedback: {detail}")
        st.markdown("**What went wrong?**")
        category = st.radio(
            "Select a category",
            [_CATEGORY_PLACEHOLDER, *AGENT_FEEDBACK_CATEGORIES],
            key=f"agent_feedback_category_{run_id}",
            label_visibility="collapsed",
        )
        comment = st.text_area(
            "Additional details (optional)",
            max_chars=500,
            key=f"agent_feedback_comment_{run_id}",
            height=80,
        )
        selected = category != _CATEGORY_PLACEHOLDER
        if st.button(
            "Submit",
            key=f"agent_feedback_down_submit_{run_id}",
            type="primary",
            width="stretch",
            disabled=not selected,
        ):
            _submit_feedback(run_id, user_id, rating="down", category=category, comment=comment)
