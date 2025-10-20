import types


def test_summary_uses_status_and_write_stream(monkeypatch):
    import pandas as pd

    from utils import chat_bot_helper as cbh

    # Prepare fake st with status and write_stream tracking
    calls = {"status": [], "write_stream": []}

    class _Ctx:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            return False

    def _status(text, expanded=False):
        calls["status"].append((text, expanded))
        return _Ctx()

    def _write_stream(gen):
        calls["write_stream"].append(True)
        # exhaust generator
        for _ in gen:
            pass

    class _State(dict):
        def __getattr__(self, k):
            try:
                return self[k]
            except KeyError:
                raise AttributeError(k)
        def __setattr__(self, k, v):
            self[k] = v

    fake_state = _State()
    fake_state.update(dict(messages=[], show_sql=True, show_table=False, show_chart=False, show_summary=True, speak_summary=False))

    fake_st = types.SimpleNamespace(
        session_state=fake_state,
        chat_message=lambda *_a, **_k: _Ctx(),
        status=_status,
        write_stream=_write_stream,
        empty=lambda: types.SimpleNamespace(markdown=lambda *_a, **_k: None, empty=lambda: None),
        rerun=lambda: None,
        error=lambda *a, **k: None,
        code=lambda *a, **k: None,
        write=lambda *a, **k: None,
        markdown=lambda *a, **k: None,
        columns=lambda sizes: [_Ctx() for _ in sizes],
        button=lambda *a, **k: False,
    )

    # VN that returns valid SQL and a tiny DF, then a one-shot summary through service fallback
    class _VN:
        def is_sql_valid(self, sql):
            return True
        def generate_sql(self, question):
            return ("SELECT 1", 0.01)
        def run_sql(self, sql):
            return (pd.DataFrame({"c":[1]}), 0.01)
        def generate_followup_questions(self, question, sql, df):
            return ["Q1"]

    # Avoid guardrails short-circuiting the flow
    monkeypatch.setattr(cbh, "st", fake_st)
    monkeypatch.setattr(cbh, "get_ethical_guideline", lambda q: ("", 1))
    # Avoid rendering side-effects in add_message
    monkeypatch.setattr(cbh, "add_message", lambda *a, **k: None)
    monkeypatch.setattr(cbh, "get_vn", lambda: _VN())
    monkeypatch.setattr(cbh, "get_summary_event_stream", lambda q, df, think=False: iter([("content", "hello "), ("content", "world")]))

    cbh.set_question("Q", render=False)
    cbh.normal_message_flow("Q")

    assert calls["status"], "st.status should be used while generating summary"
    assert calls["write_stream"], "st.write_stream should be used to display streamed content"

