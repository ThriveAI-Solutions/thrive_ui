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
            "show_summary": True,
            "speak_summary": False,
            "manual_summary_cache": {},
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
    st.warning = lambda *a, **k: None
    st.caption = lambda *a, **k: None
    st.info = lambda *a, **k: None
    st.stop = lambda: None
    st.expander = lambda *_a, **_k: nullcontext()
    st.dataframe = lambda *a, **k: None
    st.plotly_chart = lambda *a, **k: None
    st.popover = lambda *a, **k: nullcontext()
    st.text = lambda *a, **k: None
    st.rerun = lambda: None
    return st


class _VNServiceSummarySometimesEmpty:
    def __init__(self, empty_first: bool):
        self.empty_first = empty_first

    def is_sql_valid(self, sql: str) -> bool:
        return True

    def generate_sql(self, question: str):
        return ("SELECT 1", 0.01)

    def run_sql(self, sql: str):
        df = pd.DataFrame({"c": [1]})
        return (df, 0.02)

    def should_generate_chart(self, question, sql, df):
        return False

    # Non-streaming fallback used in tests below
    def generate_summary(self, question: str, df: pd.DataFrame):
        if self.empty_first:
            self.empty_first = False
            return ("", 0.01)
        return ("non-empty summary", 0.02)

    def generate_followup_questions(self, question: str, sql: str, df):
        return ["Follow-up 1", "Follow-up 2"]


def test_failed_summary_does_not_cache(monkeypatch):
    import utils.chat_bot_helper as cbh

    fake_st = _fake_st()
    monkeypatch.setattr(cbh, "st", fake_st)
    monkeypatch.setattr(cbh, "get_ethical_guideline", lambda q: ("", 1))
    monkeypatch.setattr(cbh.Message, "save", lambda self: self, raising=True)

    vn = _VNServiceSummarySometimesEmpty(empty_first=True)
    monkeypatch.setattr(cbh, "get_vn", lambda: vn)

    cbh.set_question("Q summary fail once", render=False)
    cbh.normal_message_flow("Q summary fail once")

    # Compute cache key
    df = pd.DataFrame({"c": [1]})
    key = cbh.create_summary_cache_key("Q summary fail once", df)
    assert key not in fake_st.session_state.manual_summary_cache

    # Now run again with non-empty summary
    cbh.set_question("Q summary success", render=False)
    cbh.normal_message_flow("Q summary success")

    key2 = cbh.create_summary_cache_key("Q summary success", pd.DataFrame({"c": [1]}))
    assert key2 in fake_st.session_state.manual_summary_cache
    assert fake_st.session_state.manual_summary_cache[key2][0].strip() != ""
