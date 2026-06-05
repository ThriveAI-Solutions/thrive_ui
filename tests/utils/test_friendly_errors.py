"""Tests for the friendly collapsible error UX in the chat UI.

Covers:
- render_friendly_error uses an expander, never st.error (no red wall).
- The active SQL error message renders an in-card Retry button; older error
  messages in history do not.
- A SQL failure no longer calls st.stop() — control returns cleanly so the
  chat input keeps working for new questions.
- A surprise exception in the chat flow is caught by the top-level safety
  net and persisted as a friendly ERROR Message.
"""

from __future__ import annotations

import types
from contextlib import nullcontext

import pandas as pd


def _fake_st():
    """Recording fake-Streamlit used to assert what the UI rendered."""
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
    st.session_state.update(
        {
            "messages": [],
            "show_sql": True,
            "show_table": True,
            "show_chart": False,
            "show_summary": False,
            "speak_summary": False,
        }
    )

    st.chat_message = lambda *_a, **_k: nullcontext()

    class _Placeholder:
        def markdown(self, *a, **k):
            pass

        def empty(self):
            pass

    st.empty = lambda: _Placeholder()

    class _Ctx:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    st.columns = lambda sizes: [_Ctx() for _ in sizes]

    # Recorders: capture each UI element the renderer emits.
    st._expander_calls = []
    st._error_calls = []
    st._warning_calls = []
    st._button_calls = []
    st._code_calls = []
    st._markdown_calls = []

    def _expander(label, *_a, **_k):
        st._expander_calls.append(label)
        return _Ctx()

    st.expander = _expander
    st.error = lambda *a, **k: st._error_calls.append(a[0] if a else None)
    st.warning = lambda *a, **k: st._warning_calls.append(a[0] if a else None)

    def _button(label, *_a, **_k):
        st._button_calls.append(label)
        return False

    st.button = _button
    st.code = lambda *a, **k: st._code_calls.append(a[0] if a else None)
    st.markdown = lambda *a, **k: st._markdown_calls.append(a[0] if a else None)
    st.write = lambda *a, **k: None
    st.caption = lambda *a, **k: None
    st.info = lambda *a, **k: None
    st.text = lambda *a, **k: None
    st.text_input = lambda *a, **k: ""

    class StopException(Exception):
        pass

    st.StopException = StopException

    def _stop():
        raise StopException()

    st.stop = _stop
    st.dataframe = lambda *a, **k: None
    st.plotly_chart = lambda *a, **k: None
    st.popover = lambda *a, **k: nullcontext()
    st.rerun = lambda: None
    return st


def test_render_friendly_error_uses_expander_not_red_wall(monkeypatch):
    """The helper should never call st.error — that's what produces the red wall."""
    import utils.chat_bot_helper as cbh

    fake_st = _fake_st()
    monkeypatch.setattr(cbh, "st", fake_st)

    cbh.render_friendly_error("syntax error near LIMIT")

    assert fake_st._error_calls == [], "render_friendly_error must not call st.error"
    assert fake_st._expander_calls, "render_friendly_error must wrap content in an expander"
    assert "Well this is awkward" in fake_st._expander_calls[0]
    assert any("syntax error near LIMIT" in (c or "") for c in fake_st._code_calls)


def test_render_friendly_error_omits_retry_by_default(monkeypatch):
    """Default rendering has no Retry button — only SQL errors opt in."""
    import utils.chat_bot_helper as cbh

    fake_st = _fake_st()
    monkeypatch.setattr(cbh, "st", fake_st)

    cbh.render_friendly_error("something broke")

    assert "Retry" not in fake_st._button_calls


def test_render_friendly_error_with_retry_button(monkeypatch):
    """When show_retry=True, the helper renders a Retry button + failed SQL."""
    import utils.chat_bot_helper as cbh

    fake_st = _fake_st()
    monkeypatch.setattr(cbh, "st", fake_st)

    cbh.render_friendly_error(
        "permission denied",
        failed_sql="SELECT * FROM secret",
        show_retry=True,
    )

    assert "Retry" in fake_st._button_calls
    assert any("SELECT * FROM secret" in (c or "") for c in fake_st._code_calls)


def test_render_error_shows_retry_only_for_active_error(monkeypatch):
    """A historical ERROR whose SQL doesn't match last_failed_sql shouldn't get a Retry button."""
    import utils.chat_bot_helper as cbh
    from orm.models import Message
    from utils.enums import MessageType, RoleType

    fake_st = _fake_st()
    monkeypatch.setattr(cbh, "st", fake_st)
    monkeypatch.setattr(cbh.Message, "save", lambda self: self, raising=True)

    fake_st.session_state["pending_sql_error"] = True
    fake_st.session_state["last_failed_sql"] = "SELECT current_failed"
    fake_st.session_state["last_run_sql_error"] = "boom"

    active = Message(RoleType.ASSISTANT, "boom", MessageType.ERROR, "SELECT current_failed", "Q", None, None)
    stale = Message(RoleType.ASSISTANT, "old", MessageType.ERROR, "SELECT old_failed", "Q-old", None, None)

    cbh._render_error(active, 0)
    cbh._render_error(stale, 1)

    # Active error: Retry button present.
    # Stale error: rendered, but its retry button was suppressed.
    assert fake_st._button_calls.count("Retry") == 1


def test_render_error_no_retry_when_no_pending_error(monkeypatch):
    """Even if message.query is set, no Retry button without an active session error."""
    import utils.chat_bot_helper as cbh
    from orm.models import Message
    from utils.enums import MessageType, RoleType

    fake_st = _fake_st()
    monkeypatch.setattr(cbh, "st", fake_st)
    monkeypatch.setattr(cbh.Message, "save", lambda self: self, raising=True)

    fake_st.session_state["pending_sql_error"] = False
    msg = Message(RoleType.ASSISTANT, "old error", MessageType.ERROR, "SELECT 1", "Q", None, None)

    cbh._render_error(msg, 0)

    assert "Retry" not in fake_st._button_calls


class _SqlFailsForever:
    """Mock VannaService whose run_sql never returns a DataFrame."""

    def is_sql_valid(self, sql):
        return True

    def generate_sql(self, question):
        return ("SELECT 1", 0.01)

    def generate_sql_retry(self, question, failed_sql=None, error_message=None, attempt_number=2, user_feedback=None):
        return ("SELECT 1 /* retry */", 0.01)

    def run_sql(self, sql):
        return ("not-a-df", 0.01)


def test_sql_failure_no_longer_calls_st_stop(monkeypatch):
    """The whole point of the fix: a SQL failure must NOT halt the script."""
    import utils.chat_bot_helper as cbh

    fake_st = _fake_st()
    monkeypatch.setattr(cbh, "st", fake_st)
    monkeypatch.setattr(cbh, "get_ethical_guideline", lambda q: ("", 1))
    monkeypatch.setattr(cbh.Message, "save", lambda self: self, raising=True)
    monkeypatch.setattr(cbh, "get_vn", lambda: _SqlFailsForever())

    cbh.set_question("Q that fails", render=False)

    # Should NOT raise StopException.
    cbh.normal_message_flow("Q that fails")

    # A friendly ERROR message landed in history…
    assert any(getattr(m, "type", None) == "error" for m in fake_st.session_state["messages"])
    # …and pending_sql_error was set so the in-history card shows Retry on its next render.
    assert fake_st.session_state.get("pending_sql_error") is True


class _RaisesRuntimeError:
    """Mock VannaService that raises a surprise exception."""

    def is_sql_valid(self, sql):
        return True

    def generate_sql(self, question):
        raise RuntimeError("kaboom from generate_sql")


def test_top_level_safety_net_catches_unexpected_exception(monkeypatch):
    """A surprise RuntimeError must surface as a friendly ERROR message, not propagate."""
    import utils.chat_bot_helper as cbh

    fake_st = _fake_st()
    monkeypatch.setattr(cbh, "st", fake_st)
    monkeypatch.setattr(cbh, "get_ethical_guideline", lambda q: ("", 1))
    monkeypatch.setattr(cbh.Message, "save", lambda self: self, raising=True)
    monkeypatch.setattr(cbh, "get_vn", lambda: _RaisesRuntimeError())

    cbh.set_question("Q that explodes", render=False)
    cbh.normal_message_flow("Q that explodes")  # must not raise

    error_msgs = [m for m in fake_st.session_state["messages"] if getattr(m, "type", None) == "error"]
    assert error_msgs, "Expected a friendly ERROR message to be persisted"
    assert "kaboom from generate_sql" in error_msgs[-1].content
    # my_question gets cleared so the same failing question doesn't re-fire on rerun.
    assert fake_st.session_state.get("my_question") is None


def test_top_level_safety_net_reraises_streamlit_control_exceptions(monkeypatch):
    """Streamlit's RerunException / StopException must still propagate so the runtime can act."""
    import utils.chat_bot_helper as cbh

    fake_st = _fake_st()
    monkeypatch.setattr(cbh, "st", fake_st)
    monkeypatch.setattr(cbh, "get_ethical_guideline", lambda q: ("", 1))
    monkeypatch.setattr(cbh.Message, "save", lambda self: self, raising=True)

    class StopException(Exception):
        pass

    StopException.__name__ = "StopException"

    class _StopVn:
        def is_sql_valid(self, sql):
            return True

        def generate_sql(self, question):
            raise StopException()

    monkeypatch.setattr(cbh, "get_vn", lambda: _StopVn())
    cbh.set_question("Q", render=False)

    raised = False
    try:
        cbh.normal_message_flow("Q")
    except StopException:
        raised = True
    assert raised, "StopException must propagate, not be swallowed by the safety net"


def test_render_error_short_content_still_uses_expander(monkeypatch):
    """The legacy short/long branching is gone — every error uses the friendly expander."""
    import utils.chat_bot_helper as cbh
    from orm.models import Message
    from utils.enums import MessageType, RoleType

    fake_st = _fake_st()
    monkeypatch.setattr(cbh, "st", fake_st)
    monkeypatch.setattr(cbh.Message, "save", lambda self: self, raising=True)

    short = Message(RoleType.ASSISTANT, "oops", MessageType.ERROR, "", "Q", None, None)
    cbh._render_error(short, 0)

    assert fake_st._warning_calls == [], "We removed the st.warning fallback for short messages"
    assert fake_st._expander_calls, "Short ERRORs should still be wrapped in the friendly expander"


def test_handle_sql_retry_click_sets_retry_context(monkeypatch):
    """Clicking Retry must arm use_retry_context with the pending error info."""
    import utils.chat_bot_helper as cbh

    fake_st = _fake_st()
    monkeypatch.setattr(cbh, "st", fake_st)

    fake_st.session_state.update(
        {
            "pending_sql_error": True,
            "pending_question": "Show me data",
            "last_failed_sql": "SELECT 1",
            "last_run_sql_error": "syntax error",
            "retry_feedback_active": "try a JOIN",
        }
    )

    cbh.handle_sql_retry_click()

    assert fake_st.session_state.get("use_retry_context") is True
    assert fake_st.session_state.get("retry_failed_sql") == "SELECT 1"
    assert fake_st.session_state.get("retry_error_msg") == "syntax error"
    assert fake_st.session_state.get("retry_user_feedback") == "try a JOIN"
    assert fake_st.session_state.get("my_question") == "Show me data"
    assert fake_st.session_state.get("pending_sql_error") is False


def test_chart_failure_includes_stashed_error_detail(monkeypatch):
    """When a chart fails, the friendly ERROR message includes the actual error from session state."""
    import utils.chat_bot_helper as cbh
    from utils.enums import MessageType

    fake_st = _fake_st()
    monkeypatch.setattr(cbh, "st", fake_st)
    monkeypatch.setattr(cbh.Message, "save", lambda self: self, raising=True)
    # get_chart prefers the module-level cbh.vn over get_vn() — null it so other tests
    # that populated it can't leak in.
    monkeypatch.setattr(cbh, "vn", None)

    class _ChartFailsVN:
        def should_generate_chart(self, question, sql, df):
            return True

        def generate_plotly_code(self, question, sql, df):
            return None, 0  # simulates the swallowed-exception path; detail is in session state

    monkeypatch.setattr(cbh, "get_vn", lambda: _ChartFailsVN())

    fake_st.session_state["last_chart_error"] = "boom: chart code blew up"

    cbh.get_chart("Q", "SELECT 1", pd.DataFrame({"a": [1, 2]}))

    errors = [m for m in fake_st.session_state["messages"] if getattr(m, "type", None) == MessageType.ERROR.value]
    assert errors, "expected a friendly chart ERROR message"
    assert "boom: chart code blew up" in errors[-1].content
    # And the key is consumed so it doesn't leak into the next flow.
    assert fake_st.session_state.get("last_chart_error") is None


def test_chart_failure_without_stashed_error_uses_generic_text(monkeypatch):
    """No detail stashed → keep the existing generic message rather than dangling a stray separator."""
    import utils.chat_bot_helper as cbh
    from utils.enums import MessageType

    fake_st = _fake_st()
    monkeypatch.setattr(cbh, "st", fake_st)
    monkeypatch.setattr(cbh.Message, "save", lambda self: self, raising=True)
    monkeypatch.setattr(cbh, "vn", None)

    class _ChartSilentNoneVN:
        def should_generate_chart(self, question, sql, df):
            return False

    monkeypatch.setattr(cbh, "get_vn", lambda: _ChartSilentNoneVN())

    cbh.get_chart("Q", "SELECT 1", pd.DataFrame({"a": [1, 2]}))

    errors = [m for m in fake_st.session_state["messages"] if getattr(m, "type", None) == MessageType.ERROR.value]
    assert errors
    assert errors[-1].content == "I was unable to generate a chart for this question."


def test_followup_failure_emits_friendly_error(monkeypatch):
    """A followup-generation failure surfaces as an ERROR message, not a silent empty list."""
    import utils.chat_bot_helper as cbh
    from utils.enums import MessageType

    fake_st = _fake_st()
    monkeypatch.setattr(cbh, "st", fake_st)
    monkeypatch.setattr(cbh.Message, "save", lambda self: self, raising=True)

    class _FollowupFailsVN:
        def generate_followup_questions(self, question, sql, df):
            return []  # vanna_calls swallowed; detail in session state

    monkeypatch.setattr(cbh, "get_vn", lambda: _FollowupFailsVN())

    fake_st.session_state["last_followup_error"] = "kaboom: followup model timeout"

    cbh.get_followup_questions("Q", "SELECT 1", pd.DataFrame({"a": [1, 2]}))

    # Should be an ERROR message, NOT a FOLLOWUP message.
    types_seen = [getattr(m, "type", None) for m in fake_st.session_state["messages"]]
    assert MessageType.ERROR.value in types_seen
    assert MessageType.FOLLOWUP.value not in types_seen
    err_msg = next(m for m in fake_st.session_state["messages"] if m.type == MessageType.ERROR.value)
    assert "kaboom: followup model timeout" in err_msg.content


def test_followup_success_still_emits_followup(monkeypatch):
    """No stashed error → the existing FOLLOWUP message path is preserved."""
    import utils.chat_bot_helper as cbh
    from utils.enums import MessageType

    fake_st = _fake_st()
    monkeypatch.setattr(cbh, "st", fake_st)
    monkeypatch.setattr(cbh.Message, "save", lambda self: self, raising=True)

    class _FollowupOkVN:
        def generate_followup_questions(self, question, sql, df):
            return ["Question A?", "Question B?"]

    monkeypatch.setattr(cbh, "get_vn", lambda: _FollowupOkVN())
    cbh.get_followup_questions("Q", "SELECT 1", pd.DataFrame({"a": [1, 2]}))

    types_seen = [getattr(m, "type", None) for m in fake_st.session_state["messages"]]
    assert MessageType.FOLLOWUP.value in types_seen
    assert MessageType.ERROR.value not in types_seen


def test_set_question_clears_optional_step_error_keys(monkeypatch):
    """Stale chart/summary/followup error stashes don't leak into the next question."""
    import utils.chat_bot_helper as cbh

    fake_st = _fake_st()
    monkeypatch.setattr(cbh, "st", fake_st)
    monkeypatch.setattr(cbh.Message, "save", lambda self: self, raising=True)

    fake_st.session_state["last_chart_error"] = "stale chart error"
    fake_st.session_state["last_summary_error"] = "stale summary error"
    fake_st.session_state["last_followup_error"] = "stale followup error"

    cbh.set_question("a fresh question", render=False)

    assert fake_st.session_state.get("last_chart_error") is None
    assert fake_st.session_state.get("last_summary_error") is None
    assert fake_st.session_state.get("last_followup_error") is None


def test_vanna_calls_stashes_chart_error(monkeypatch):
    """The helper in vanna_calls actually writes the error into session state on failure."""
    import streamlit as st_real

    import utils.vanna_calls as vc

    captured = {}

    class _FakeState(dict):
        def __setitem__(self, k, v):
            captured[k] = v
            super().__setitem__(k, v)

    monkeypatch.setattr(st_real, "session_state", _FakeState())

    vc._stash_optional_step_error("last_chart_error", RuntimeError("plot crashed"))

    assert captured.get("last_chart_error") == "plot crashed"


def test_new_question_after_error_clears_pending_state(monkeypatch):
    """After a SQL error, typing a new question proceeds without needing to click Retry."""
    import utils.chat_bot_helper as cbh

    fake_st = _fake_st()
    monkeypatch.setattr(cbh, "st", fake_st)
    monkeypatch.setattr(cbh, "get_ethical_guideline", lambda q: ("", 1))
    monkeypatch.setattr(cbh.Message, "save", lambda self: self, raising=True)

    class _FailThenSucceed:
        def __init__(self):
            self.failed_once = False

        def is_sql_valid(self, sql):
            return True

        def generate_sql(self, question):
            return ("SELECT 1", 0.01)

        def generate_sql_retry(
            self, question, failed_sql=None, error_message=None, attempt_number=2, user_feedback=None
        ):
            return ("SELECT 1 /* retry */", 0.01)

        def run_sql(self, sql):
            if not self.failed_once:
                self.failed_once = True
                return ("not-a-df", 0.01)
            return (pd.DataFrame({"c": [1]}), 0.02)

        def should_generate_chart(self, question, sql, df):
            return False

        def generate_summary(self, question, df):
            return ("ok", 0.01)

        def generate_followup_questions(self, question, sql, df):
            return []

    monkeypatch.setattr(cbh, "get_vn", lambda: _FailThenSucceed())

    cbh.set_question("Q1 fails", render=False)
    cbh.normal_message_flow("Q1 fails")  # must not raise
    assert fake_st.session_state.get("pending_sql_error") is True

    cbh.set_question("Q2 works", render=False)
    # set_question alone is supposed to clear the lingering error state.
    assert fake_st.session_state.get("pending_sql_error") is False
    assert fake_st.session_state.get("last_failed_sql") in (None, "")
    assert fake_st.session_state.get("last_run_sql_error") in (None, "")
