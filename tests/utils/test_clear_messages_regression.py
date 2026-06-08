"""Regression test for issue #113.

When the user clicked the sidebar **Clear** button or typed the `/clear` magic
command, `set_question(None)` set `st.session_state.messages = None`. On the
very next render, `get_unique_messages()` crashed with:

    TypeError: 'NoneType' object is not iterable

The fix changes the reset value to an empty list. This test pins the contract
so it cannot silently regress to `None`.
"""

import types
from contextlib import nullcontext


def _fake_st_with_message():
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
    fake_msg = types.SimpleNamespace(id=42, role="user", content="hello")
    st.session_state.update({"messages": [fake_msg], "min_message_id": 0})

    st.chat_message = lambda *_a, **_k: nullcontext()
    st.empty = lambda: None
    st.rerun = lambda: None
    st.toast = lambda *_a, **_k: None
    st.markdown = lambda *_a, **_k: None
    return st


def test_set_question_none_leaves_messages_as_empty_list(monkeypatch):
    import utils.chat_bot_helper as cbh

    fake_st = _fake_st_with_message()
    monkeypatch.setattr(cbh, "st", fake_st)
    monkeypatch.setattr(cbh, "save_user_settings", lambda: None)

    cbh.set_question(None)

    # Must be an empty list, not None — otherwise get_unique_messages crashes
    assert fake_st.session_state.messages == []
    assert fake_st.session_state.messages is not None
    assert fake_st.session_state.my_question is None
    assert fake_st.session_state.min_message_id == 42


def test_get_unique_messages_after_clear_does_not_crash(monkeypatch):
    """End-to-end: clearing then calling get_unique_messages must not raise."""
    import utils.chat_bot_helper as cbh

    fake_st = _fake_st_with_message()
    monkeypatch.setattr(cbh, "st", fake_st)
    monkeypatch.setattr(cbh, "save_user_settings", lambda: None)

    cbh.set_question(None)

    # Before the fix this raised TypeError: 'NoneType' object is not iterable
    result = cbh.get_unique_messages()
    assert result == []
