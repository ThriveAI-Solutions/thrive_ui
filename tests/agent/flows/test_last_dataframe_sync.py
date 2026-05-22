"""After a streaming run completes, the runner must mirror
deps.last_dataframe to st.session_state["df"] so the Vanna magic-function
slash commands operate on the agent's most recent result without
modification.

Per Phase 3 design §3.2.
"""

from __future__ import annotations
import pandas as pd

from agent.runner import _sync_last_dataframe_to_session_state


def test_sync_writes_dataframe_to_session_state_df_key():
    deps_dataframe = pd.DataFrame({"a": [1, 2, 3]})
    fake_session_state: dict = {}

    _sync_last_dataframe_to_session_state(deps_dataframe, fake_session_state)

    assert "df" in fake_session_state
    assert fake_session_state["df"].equals(deps_dataframe)


def test_sync_with_none_leaves_session_state_df_untouched():
    """When the tool turn produces no DataFrame, do NOT touch
    session_state["df"]. A prior turn's df should survive so the user
    can still slash-command against it."""
    fake_session_state: dict = {"df": pd.DataFrame({"a": [1]})}

    _sync_last_dataframe_to_session_state(None, fake_session_state)

    assert "df" in fake_session_state
    assert len(fake_session_state["df"]) == 1


def test_sync_with_empty_dataframe_still_writes():
    """An explicit empty DataFrame IS a valid result (e.g., 'no
    encounters found'); sync should overwrite. Only None means
    'this turn didn't produce a dataframe at all'."""
    empty_df = pd.DataFrame()
    fake_session_state: dict = {"df": pd.DataFrame({"a": [1]})}

    _sync_last_dataframe_to_session_state(empty_df, fake_session_state)

    assert "df" in fake_session_state
    assert len(fake_session_state["df"]) == 0
