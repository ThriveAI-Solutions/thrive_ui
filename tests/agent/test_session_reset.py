"""Unit tests for agent.session_reset.

The helper operates on a dict-like session_state (Streamlit's
st.session_state quacks like a dict for our purposes), so these
tests pass a plain dict and verify pop/set behavior directly.
"""

import pytest

from agent.session_reset import RESET_KEYS, reset_agent_session


def _populated_session():
    """A session_state-like dict pre-populated with every key we expect
    to clear, plus a representative set of keys we expect to preserve."""
    return {
        # --- keys that MUST be cleared ---
        "agent_session_id": "abc-123",
        "agent_message_history": [{"role": "user", "content": "hi"}],
        "pending_user_question": "what's the count?",
        "my_question": "what's the count?",
        "df": object(),
        "last_sql": "select 1",
        "last_run_sql_error": "boom",
        "last_failed_sql": "select bad",
        "pending_sql_error": True,
        "retry_failed_sql": "select bad",
        "streamed_summary": "summary text",
        "streamed_summary_for_question": "what's the count?",
        "streamed_summary_elapsed_time": 1.23,
        "_vn_instance": object(),
        "_pending_agent_reset_at": 1715600000.0,
        # messages is special — set to [], not popped
        "messages": [{"role": "user", "content": "hi"}],
        # --- keys that MUST be preserved ---
        "cookies": object(),
        "user": object(),
        "user_id": 42,
        "user_role": 1,
        "user_theme": "welltellai",
        "show_chart": False,
        "show_elapsed_time": True,
        "show_followup": False,
        "show_question_history": True,
        "show_sql": True,
        "show_suggested": False,
        "show_table": True,
        "speak_summary": False,
        "voice_input": False,
        "agentic_mode": True,
        "llm_fallback": False,
        "confirm_magic_commands": True,
        "selected_llm_provider": "anthropic",
        "selected_llm_model": "claude-3-5-sonnet-latest",
        "selected_patient_source_id": "pt-9",
        "selected_patient_display_name": "Jane Doe",
        "selected_patient_dob": "1980-01-01",
        "selection_origin": "user_click",
        "selected_at": "2026-05-13T10:00:00",
        "community_questions": ["q1", "q2"],
        "_messages_loaded": True,
    }


def test_reset_pops_every_reset_key():
    session = _populated_session()
    reset_agent_session(session)
    for key in RESET_KEYS:
        assert key not in session, f"{key} should have been popped"


def test_reset_sets_messages_to_empty_list():
    session = _populated_session()
    reset_agent_session(session)
    assert session["messages"] == []


PRESERVED_KEYS = (
    "cookies",
    "user",
    "user_id",
    "user_role",
    "user_theme",
    "show_chart",
    "show_elapsed_time",
    "show_followup",
    "show_question_history",
    "show_sql",
    "show_suggested",
    "show_table",
    "speak_summary",
    "voice_input",
    "agentic_mode",
    "llm_fallback",
    "confirm_magic_commands",
    "selected_llm_provider",
    "selected_llm_model",
    "selected_patient_source_id",
    "selected_patient_display_name",
    "selected_patient_dob",
    "selection_origin",
    "selected_at",
    "community_questions",
    "_messages_loaded",
)


@pytest.mark.parametrize("key", PRESERVED_KEYS)
def test_reset_preserves_non_agent_keys(key):
    session = _populated_session()
    original = session[key]
    reset_agent_session(session)
    assert key in session, f"{key} should have been preserved"
    assert session[key] is original or session[key] == original


def test_reset_on_empty_dict_is_no_op_except_messages():
    session = {}
    reset_agent_session(session)
    # The reset always normalizes messages to []
    assert session == {"messages": []}


def test_reset_is_idempotent():
    session = _populated_session()
    reset_agent_session(session)
    snapshot = dict(session)
    reset_agent_session(session)
    assert session == snapshot


def test_reset_keys_includes_pending_reset_flag():
    """The arming flag itself must be in the pop list — otherwise the
    confirm UI would stay 'armed' across a successful reset."""
    assert "_pending_agent_reset_at" in RESET_KEYS


def test_messages_not_in_reset_keys():
    """The `messages` key is set to [] by reset_agent_session, not popped.
    Adding it to RESET_KEYS would invert that semantics."""
    assert "messages" not in RESET_KEYS
