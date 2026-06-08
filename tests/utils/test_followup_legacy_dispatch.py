"""Regression tests for issue #114.

`FOLLOW_UP_MAGIC_RENDERERS` was unreachable in non-agentic mode because
`views/chat_bot.py` only forwarded a `previous_df` when `agentic_mode=True`.
The fix:

1. Adds `get_last_assistant_dataframe()` to recover the most recent dataframe
   from session-state message history.
2. Restructures `is_magic_do_magic()` so the main MAGIC_RENDERERS and the
   FOLLOW_UP_MAGIC_RENDERERS are checked independently — the slash commands
   are no longer suppressed when a `previous_df` is supplied.

These tests pin the new dispatch contract.
"""

import types
from contextlib import nullcontext

import pandas as pd


def _fake_st(messages=None):
    st = types.SimpleNamespace()

    class _State(dict):
        def __getattr__(self, k):
            try:
                return self[k]
            except KeyError:
                raise AttributeError(k)

        def __setattr__(self, k, v):
            self[k] = v

    st.session_state = _State()
    st.session_state.update({"messages": messages or [], "agentic_mode": False})

    # Make st.secrets behave like a dict supporting `.get(key, default)`.
    st.secrets = {"features": {"semantic_magic_enabled": False}}

    st.chat_message = lambda *_a, **_k: nullcontext()
    st.empty = lambda: None
    st.rerun = lambda: None
    st.stop = lambda: None
    st.markdown = lambda *_a, **_k: None
    st.toast = lambda *_a, **_k: None
    st.write = lambda *_a, **_k: None
    return st


class _FakeMessage:
    def __init__(self, role, dataframe=None):
        self.role = role
        self.dataframe = dataframe


# ---------------- get_last_assistant_dataframe ----------------


def test_get_last_assistant_dataframe_returns_most_recent(monkeypatch):
    import utils.chat_bot_helper as cbh

    df_old = pd.DataFrame({"a": [1]})
    df_new = pd.DataFrame({"b": [2, 3]})
    messages = [
        _FakeMessage("user"),
        _FakeMessage("assistant", dataframe=df_old.to_json(date_format="iso")),
        _FakeMessage("user"),
        _FakeMessage("assistant", dataframe=df_new.to_json(date_format="iso")),
        _FakeMessage("assistant"),  # no dataframe, ignored
    ]
    fake_st = _fake_st(messages)
    monkeypatch.setattr(cbh, "st", fake_st)

    out = cbh.get_last_assistant_dataframe()
    assert out is not None
    assert list(out.columns) == ["b"]
    assert out["b"].tolist() == [2, 3]


def test_get_last_assistant_dataframe_returns_none_when_no_history(monkeypatch):
    import utils.chat_bot_helper as cbh

    monkeypatch.setattr(cbh, "st", _fake_st([]))
    assert cbh.get_last_assistant_dataframe() is None


def test_get_last_assistant_dataframe_returns_none_on_corrupt_most_recent(monkeypatch):
    """If the newest assistant df is corrupt, return None — never silently fall back
    to an older frame, which would mislead the caller into running follow-up commands
    against stale data."""
    import utils.chat_bot_helper as cbh

    df_old = pd.DataFrame({"a": [1, 2, 3]})
    messages = [
        _FakeMessage("assistant", dataframe=df_old.to_json(date_format="iso")),
        _FakeMessage("user"),
        _FakeMessage("assistant", dataframe="not-valid-json{["),  # newest, corrupt
    ]
    fake_st = _fake_st(messages)
    monkeypatch.setattr(cbh, "st", fake_st)

    assert cbh.get_last_assistant_dataframe() is None


def test_get_last_assistant_dataframe_tolerates_messages_none(monkeypatch):
    """Defensive: session_state.messages can legitimately be missing at boot."""
    import utils.chat_bot_helper as cbh

    fake_st = _fake_st(None)
    # Don't seed messages at all
    fake_st.session_state.pop("messages", None)
    monkeypatch.setattr(cbh, "st", fake_st)

    assert cbh.get_last_assistant_dataframe() is None


# ---------------- is_magic_do_magic dispatch ----------------


def test_slash_command_still_works_when_previous_df_set(monkeypatch):
    """Regression: providing a previous_df must not suppress MAGIC_RENDERERS."""
    import utils.magic_functions as mf

    monkeypatch.setattr(mf, "st", _fake_st())
    monkeypatch.setattr(mf, "add_message", lambda *_a, **_k: None)
    monkeypatch.setattr(mf, "get_current_group_id", lambda: None)

    calls = []

    def fake_help(question, params, previous_df):
        calls.append(("help", question, previous_df))

    # Patch the bound function via the registry, not by name
    original = mf.MAGIC_RENDERERS[r"^/help$"]["func"]
    mf.MAGIC_RENDERERS[r"^/help$"]["func"] = fake_help
    try:
        df = pd.DataFrame({"x": [1, 2]})
        handled = mf.is_magic_do_magic("/help", previous_df=df)
    finally:
        mf.MAGIC_RENDERERS[r"^/help$"]["func"] = original

    assert handled is True
    assert calls == [("help", "/help", None)], (
        "Slash command must route to MAGIC_RENDERERS with previous_df=None even when "
        "a previous dataframe is supplied"
    )


def test_bare_followup_command_dispatches_when_previous_df_set(monkeypatch):
    """`head 3` should route to FOLLOW_UP_MAGIC_RENDERERS when previous_df is set."""
    import utils.magic_functions as mf

    monkeypatch.setattr(mf, "st", _fake_st())
    monkeypatch.setattr(mf, "add_message", lambda *_a, **_k: None)
    monkeypatch.setattr(mf, "get_current_group_id", lambda: None)

    calls = []

    def fake_head(question, params, previous_df):
        calls.append((question, params, previous_df))

    pattern = r"^head\s+(?P<num_rows>\d+)$"
    original = mf.FOLLOW_UP_MAGIC_RENDERERS[pattern]["func"]
    mf.FOLLOW_UP_MAGIC_RENDERERS[pattern]["func"] = fake_head
    try:
        df = pd.DataFrame({"x": [1, 2, 3]})
        handled = mf.is_magic_do_magic("head 3", previous_df=df)
    finally:
        mf.FOLLOW_UP_MAGIC_RENDERERS[pattern]["func"] = original

    assert handled is True
    assert len(calls) == 1
    question, params, df_arg = calls[0]
    assert question == "head 3"
    assert params == {"num_rows": "3"}
    assert df_arg is df, "FOLLOW_UP handler must receive the supplied previous_df"


def test_bare_followup_command_ignored_when_no_previous_df(monkeypatch):
    """`head 3` with no prior dataframe must NOT match — falls through to LLM."""
    import utils.magic_functions as mf

    monkeypatch.setattr(mf, "st", _fake_st())
    monkeypatch.setattr(mf, "add_message", lambda *_a, **_k: None)
    monkeypatch.setattr(mf, "get_current_group_id", lambda: None)

    calls = []
    pattern = r"^head\s+(?P<num_rows>\d+)$"
    original = mf.FOLLOW_UP_MAGIC_RENDERERS[pattern]["func"]
    mf.FOLLOW_UP_MAGIC_RENDERERS[pattern]["func"] = lambda *a, **k: calls.append(a)
    try:
        handled = mf.is_magic_do_magic("head 3", previous_df=None)
    finally:
        mf.FOLLOW_UP_MAGIC_RENDERERS[pattern]["func"] = original

    assert handled is False
    assert calls == []


def test_slash_command_without_previous_df_still_works(monkeypatch):
    """Baseline guard — the legacy non-agentic slash-command path is unchanged."""
    import utils.magic_functions as mf

    monkeypatch.setattr(mf, "st", _fake_st())
    monkeypatch.setattr(mf, "add_message", lambda *_a, **_k: None)
    monkeypatch.setattr(mf, "get_current_group_id", lambda: None)

    calls = []
    original = mf.MAGIC_RENDERERS[r"^/help$"]["func"]
    mf.MAGIC_RENDERERS[r"^/help$"]["func"] = lambda q, p, df: calls.append((q, df))
    try:
        handled = mf.is_magic_do_magic("/help", previous_df=None)
    finally:
        mf.MAGIC_RENDERERS[r"^/help$"]["func"] = original

    assert handled is True
    assert calls == [("/help", None)]
