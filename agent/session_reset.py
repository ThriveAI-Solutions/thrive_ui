"""One-shot reset of agent-side conversation state.

Used by the "Reset agent" sidebar button in views/chat_bot.py. The
helper is intentionally Streamlit-free: it takes a dict-like
session_state and mutates it in place. The caller is responsible for
calling st.rerun().

Preserves: user identity (cookies, user, user_id, user_role), theme,
sidebar preferences (show_*, agentic_mode, etc.), LLM selection, the
selected_patient slot, and the _messages_loaded sentinel that suppresses
the page-load rehydrate-from-SQLite behavior after a reset.
"""

from __future__ import annotations

from typing import MutableMapping

RESET_KEYS: tuple[str, ...] = (
    # agent conversation / in-flight work
    "agent_session_id",
    "agent_message_history",
    "pending_user_question",
    "my_question",
    # last query state
    "df",
    "last_sql",
    "last_run_sql_error",
    "last_failed_sql",
    "pending_sql_error",
    "retry_failed_sql",
    "streamed_summary",
    "streamed_summary_for_question",
    "streamed_summary_elapsed_time",
    # vanna cache
    "_vn_instance",
    # confirm-button arming flag (popped so the UI returns to Idle)
    "_pending_agent_reset_at",
)


def reset_agent_session(session_state: MutableMapping) -> None:
    """Pop every key in RESET_KEYS from session_state, then set messages
    to an empty list. Missing keys are silently ignored."""
    for key in RESET_KEYS:
        session_state.pop(key, None)
    session_state["messages"] = []
