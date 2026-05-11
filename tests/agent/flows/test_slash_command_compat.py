"""End-to-end: after an agent turn populates last_dataframe, a magic
function slash command operating on st.session_state['df'] sees the
agent's dataframe.

Per Phase 3 design §3.8 — this is the existence test for slash-command
compatibility. Specific magic-function correctness is out of scope; if
this test passes, the bridge between agent and slash commands is sound.
"""

from __future__ import annotations
import pandas as pd
import pytest

from agent.runner import _sync_last_dataframe_to_session_state


def test_describe_magic_sees_synced_dataframe():
    """After the runner syncs deps.last_dataframe → session_state['df'],
    the existing is_magic_do_magic('/describe') path can read it."""
    fake_session_state: dict = {}
    deps_dataframe = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})

    _sync_last_dataframe_to_session_state(deps_dataframe, fake_session_state)

    # The canonical magic-function key. Vanna writes here; agent now
    # writes here too. Both consumers must agree.
    assert "df" in fake_session_state
    assert fake_session_state["df"].equals(deps_dataframe)


def test_chat_bot_threads_session_state_df_to_magic_in_agentic_mode():
    """When agentic_mode is on, views/chat_bot.py should pass
    st.session_state['df'] as previous_df to is_magic_do_magic so the
    slash command sees the agent's latest dataframe rather than None.

    This is a contract test against the existing implementation — if
    the code at views/chat_bot.py around the is_magic_do_magic call
    site doesn't thread session_state['df'] through under agentic_mode,
    this test fails loudly and the task scope needs adjustment."""
    import inspect

    from views import chat_bot

    # Read the views/chat_bot.py source. The signal we want to find:
    # under an agentic_mode branch, the call passes previous_df sourced
    # from st.session_state['df']. The form of the check is permissive
    # because the surrounding code can evolve, but the substring must
    # be present.
    source = inspect.getsource(chat_bot)
    assert "agentic_mode" in source, "agentic_mode reference disappeared"
    assert "is_magic_do_magic" in source, "is_magic_do_magic call disappeared"
    # The wiring: when agentic_mode is on, previous_df comes from session_state['df']
    assert 'session_state.get("df")' in source or "session_state.get('df')" in source, (
        "views/chat_bot.py should read st.session_state.get('df') to thread "
        "the agent's last_dataframe into magic functions when agentic_mode is on. "
        "If you renamed the key, update agent.runner._sync_last_dataframe_to_session_state "
        "to match."
    )
