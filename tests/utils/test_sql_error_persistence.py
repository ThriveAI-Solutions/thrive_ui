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
    st.empty = lambda: types.SimpleNamespace()

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
    st.caption = lambda *a, **k: None
    st.stop = lambda: None
    st.expander = lambda *_a, **_k: nullcontext()
    st.dataframe = lambda *a, **k: None
    st.plotly_chart = lambda *a, **k: None
    st.popover = lambda *a, **k: nullcontext()
    st.text = lambda *a, **k: None
    st.rerun = lambda: None
    return st


class _DummyVNServiceFailThenSuccess:
    def __init__(self):
        self._first = True

    def is_sql_valid(self, sql: str) -> bool:
        return True

    def generate_sql(self, question: str):
        return ("SELECT 1", 0.01)

    def run_sql(self, sql: str):
        if self._first:
            self._first = False
            return ("not-a-df", 0.01)
        else:
            df = pd.DataFrame({"c": [1]})
            return (df, 0.02)

    def should_generate_chart(self, question, sql, df):
        return False

    def generate_summary(self, question: str, df: pd.DataFrame):
        return ("ok", 0.01)


def test_clears_pending_sql_error_after_success(monkeypatch):
    import utils.chat_bot_helper as cbh

    fake_st = _fake_st()
    monkeypatch.setattr(cbh, "st", fake_st)
    monkeypatch.setattr(cbh, "get_ethical_guideline", lambda q: ("", 1))
    monkeypatch.setattr(cbh.Message, "save", lambda self: self, raising=True)

    vn = _DummyVNServiceFailThenSuccess()
    monkeypatch.setattr(cbh, "get_vn", lambda: vn)

    # First question triggers SQL error inline, sets pending_sql_error
    cbh.set_question("Q1 fail first", render=False)
    cbh.normal_message_flow("Q1 fail first")
    assert fake_st.session_state.get("pending_sql_error", False) is True

    # New question should clear flags at start and after success
    cbh.set_question("Q2 success", render=False)
    cbh.normal_message_flow("Q2 success")

    assert fake_st.session_state.get("pending_sql_error", False) is False
    assert fake_st.session_state.get("last_run_sql_error") in (None, "")
    assert fake_st.session_state.get("last_failed_sql") in (None, "")


