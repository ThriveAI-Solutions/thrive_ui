import types
from contextlib import nullcontext

import pandas as pd


def _fake_st():
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

    st.chat_message = lambda *_args, **_kwargs: nullcontext()

    # Create a placeholder-like object with markdown and empty methods
    class _Placeholder:
        def markdown(self, *args, **kwargs):
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
    st.button = lambda *a, **k: False
    st.write = lambda *a, **k: None
    st.markdown = lambda *a, **k: None
    st.code = lambda *a, **k: None
    st.error = lambda *a, **k: None
    st.warning = lambda *a, **k: None
    st.caption = lambda *a, **k: None
    st.info = lambda *a, **k: None

    # st.stop() in real Streamlit raises StopException to halt execution
    class StopException(Exception):
        pass

    st.StopException = StopException

    def _stop():
        raise StopException()

    st.stop = _stop
    st.expander = lambda *_a, **_k: nullcontext()
    st.dataframe = lambda *a, **k: None
    st.plotly_chart = lambda *a, **k: None
    st.popover = lambda *a, **k: nullcontext()
    st.text = lambda *a, **k: None
    st.rerun = lambda: None
    return st


class _DummyVNServiceFailThenSuccess:
    """Mock VannaService that fails for the first question (all retries) then succeeds on subsequent questions."""

    def __init__(self):
        self._first_question = True
        self._fail_count = 0

    def is_sql_valid(self, sql: str) -> bool:
        return True

    def generate_sql(self, question: str):
        return ("SELECT 1", 0.01)

    def generate_sql_retry(self, question: str, failed_sql=None, error_message=None, attempt_number=2):
        """Retry method called by auto-retry loop."""
        return ("SELECT 1 /* retry */", 0.01)

    def run_sql(self, sql: str):
        # First question: fail all attempts (3 total with default 2 retries)
        if self._first_question:
            self._fail_count += 1
            if self._fail_count >= 3:  # After 3 failures, mark first question as done
                self._first_question = False
            return ("not-a-df", 0.01)
        else:
            # Subsequent questions: succeed
            df = pd.DataFrame({"c": [1]})
            return (df, 0.02)

    def should_generate_chart(self, question, sql, df):
        return False

    def generate_summary(self, question: str, df: pd.DataFrame):
        return ("ok", 0.01)

    def generate_followup_questions(self, question: str, sql: str, df):
        return ["Follow-up 1", "Follow-up 2"]


def test_clears_pending_sql_error_after_success(monkeypatch):
    import utils.chat_bot_helper as cbh

    fake_st = _fake_st()
    monkeypatch.setattr(cbh, "st", fake_st)
    monkeypatch.setattr(cbh, "get_ethical_guideline", lambda q: ("", 1))
    monkeypatch.setattr(cbh.Message, "save", lambda self: self, raising=True)

    vn = _DummyVNServiceFailThenSuccess()
    monkeypatch.setattr(cbh, "get_vn", lambda: vn)

    # First question triggers SQL error after all retries, sets pending_sql_error
    cbh.set_question("Q1 fail first", render=False)
    try:
        cbh.normal_message_flow("Q1 fail first")
    except fake_st.StopException:
        pass  # Expected - st.stop() raises this after error
    assert fake_st.session_state.get("pending_sql_error", False) is True

    # New question should clear flags at start and after success
    cbh.set_question("Q2 success", render=False)
    cbh.normal_message_flow("Q2 success")

    assert fake_st.session_state.get("pending_sql_error", False) is False
    assert fake_st.session_state.get("last_run_sql_error") in (None, "")
    assert fake_st.session_state.get("last_failed_sql") in (None, "")


def test_invalid_sql_does_not_call_llm(monkeypatch):
    import utils.chat_bot_helper as cbh

    fake_st = _fake_st()
    fake_st.session_state.update(
        {
            "show_sql": True,
            "show_table": False,
            "show_chart": False,
            "show_summary": False,
            "speak_summary": False,
        }
    )

    called_llm = {"called": False}
    monkeypatch.setattr(cbh, "st", fake_st)
    monkeypatch.setattr(cbh, "get_ethical_guideline", lambda q: ("", 1))  # Allow question through
    monkeypatch.setattr(cbh, "call_llm", lambda *_a, **_k: called_llm.__setitem__("called", True))
    monkeypatch.setattr(cbh.Message, "save", lambda self: self, raising=True)

    class _VNInvalid:
        def is_sql_valid(self, sql: str) -> bool:
            return False

        def generate_sql(self, question: str):
            return ("SELECT oops", 0.01)

        def generate_sql_retry(self, question: str, failed_sql=None, error_message=None, attempt_number=2):
            return ("SELECT oops /* retry */", 0.01)

    monkeypatch.setattr(cbh, "get_vn", lambda: _VNInvalid())

    cbh.set_question("Q invalid", render=False)
    try:
        cbh.normal_message_flow("Q invalid")
    except fake_st.StopException:
        pass  # Expected - invalid SQL triggers stop after retries

    # Ensure we posted an ERROR message and did not call LLM fallback
    assert any(getattr(m, "type", None) == "error" for m in fake_st.session_state.get("messages", []))
    assert called_llm["called"] is False


def test_no_summary_added_when_sql_error(monkeypatch):
    import utils.chat_bot_helper as cbh

    fake_st = _fake_st()
    # Ensure UI toggles would normally allow summary generation
    fake_st.session_state.update(
        {
            "show_sql": True,
            "show_table": False,
            "show_chart": False,
            "show_summary": True,
            "speak_summary": False,
        }
    )

    monkeypatch.setattr(cbh, "st", fake_st)
    monkeypatch.setattr(cbh, "get_ethical_guideline", lambda q: ("", 1))
    monkeypatch.setattr(cbh.Message, "save", lambda self: self, raising=True)

    class _FailExecSvc:
        def is_sql_valid(self, sql: str) -> bool:
            return True

        def generate_sql(self, question: str):
            return ("SELECT 1", 0.01)

        def generate_sql_retry(self, question: str, failed_sql=None, error_message=None, attempt_number=2):
            """Retry method called by auto-retry loop."""
            return ("SELECT 1 /* retry */", 0.01)

        def run_sql(self, sql: str):
            # Simulate execution error: returns non-DataFrame path
            return ("not-a-df", 0.01)

    vn = _FailExecSvc()
    monkeypatch.setattr(cbh, "get_vn", lambda: vn)

    # Trigger flow; it should render inline error and st.stop() raises StopException
    cbh.set_question("Q err", render=False)
    try:
        cbh.normal_message_flow("Q err")
    except fake_st.StopException:
        pass  # Expected - st.stop() raises this after all retries fail

    # Verify that an error is pending and no SUMMARY message exists in messages
    assert fake_st.session_state.get("pending_sql_error", False) is True
    assert all(getattr(m, "type", None) != "summary" for m in fake_st.session_state.get("messages", []))
