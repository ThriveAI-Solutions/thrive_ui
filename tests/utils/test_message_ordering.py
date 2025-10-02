import types
from contextlib import nullcontext

import pandas as pd
import pytest


def _fake_st():
    st = types.SimpleNamespace()
    # Minimal session_state required by chat flow
    # Provide a dict-like object that supports get()
    class _State(dict):
        def __getattr__(self, k):
            try:
                return self[k]
            except KeyError:
                raise AttributeError(k)
        def __setattr__(self, k, v):
            self[k] = v
    st.session_state = _State()
    st.session_state.update({
        "messages": [],
        "show_sql": True,
        "show_table": True,
        "show_chart": False,
        "show_summary": True,
        "speak_summary": False,
    })

    # UI no-ops
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
    st.toast = lambda *a, **k: None
    st.stop = lambda: None
    st.expander = lambda *_a, **_k: nullcontext()
    st.dataframe = lambda *a, **k: None
    st.plotly_chart = lambda *a, **k: None
    st.popover = lambda *a, **k: nullcontext()
    st.text = lambda *a, **k: None
    return st


def _fake_save(self):
    # Assign incremental ids and bypass database
    if not hasattr(_fake_save, "_i"):
        _fake_save._i = 0
    _fake_save._i += 1
    self.id = _fake_save._i
    return self


class _DummyUnderlying:
    def __init__(self, has_thinking: bool):
        self.ollama_client = object() if has_thinking else None


class _DummyVNService:
    def __init__(self, has_thinking: bool):
        self.vn = _DummyUnderlying(has_thinking)

    # Decision helpers
    def is_sql_valid(self, sql: str) -> bool:
        return True

    # Main pipeline
    def generate_sql(self, question: str):
        return ("SELECT 1 AS total_patients", 0.01)

    def run_sql(self, sql: str):
        df = pd.DataFrame({"total_patients": [193]})
        return (df, 0.02)

    def should_generate_chart(self, question, sql, df):
        return False

    def generate_summary(self, question: str, df: pd.DataFrame):
        return ("There are 193 patients.", 0.01)

    def generate_followup_questions(self, question: str, sql: str, df: pd.DataFrame):
        return ["Follow-up 1", "Follow-up 2"]


def _fake_thought_stream(_question: str):
    # Simulate a thinking model stream; our code collects then updates once
    yield "Analyzing..."
    yield " Done."


@pytest.mark.parametrize("has_thinking", [True, False])
def test_message_ordering_thinking_and_summary(monkeypatch, has_thinking):
    # Patch st in chat_bot_helper to a fake headless version
    import utils.chat_bot_helper as cbh

    fake_st = _fake_st()
    monkeypatch.setattr(cbh, "st", fake_st)

    # Ensure guardrails do not block
    monkeypatch.setattr(cbh, "get_ethical_guideline", lambda q: ("", 1))

    # Patch Message.save to bypass DB
    monkeypatch.setattr(cbh.Message, "save", _fake_save, raising=True)

    # Provide a deterministic Vanna service
    monkeypatch.setattr(cbh, "get_vn", lambda: _DummyVNService(has_thinking))
    # Provide a simple thought stream for thinking backends
    monkeypatch.setattr(cbh, "get_llm_sql_thought_stream", _fake_thought_stream)

    # Start with a user question persisted
    cbh.set_question("How many patients are in the wny_health table?", render=False)

    # Run the normal flow (this appends assistant messages)
    cbh.normal_message_flow("How many patients are in the wny_health table?")

    # Collect full type sequence
    full_seq = [m.type for m in fake_st.session_state.messages]
    # Find the last user text message position
    try:
        pos = len(full_seq) - 1 - full_seq[::-1].index("text")
    except ValueError:
        pos = None
    assert pos is not None, f"No user TEXT message found. seq={full_seq}"

    # Expected subsequence must begin at the user message
    if has_thinking:
        must_prefix = ["text", "thinking", "sql", "dataframe", "summary"]
    else:
        must_prefix = ["text", "sql", "dataframe", "summary"]

    assert full_seq[pos:pos+len(must_prefix)] == must_prefix, f"Got seq from pos: {full_seq[pos:pos+len(must_prefix)]} full={full_seq}"


@pytest.mark.parametrize(
    "first_thinking,second_thinking",
    [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ],
)
def test_two_questions_ordering(monkeypatch, first_thinking, second_thinking):
    import utils.chat_bot_helper as cbh

    fake_st = _fake_st()
    monkeypatch.setattr(cbh, "st", fake_st)
    monkeypatch.setattr(cbh, "get_ethical_guideline", lambda q: ("", 1))
    monkeypatch.setattr(cbh.Message, "save", _fake_save, raising=True)

    # First question
    monkeypatch.setattr(cbh, "get_vn", lambda: _DummyVNService(first_thinking))
    monkeypatch.setattr(cbh, "get_llm_sql_thought_stream", _fake_thought_stream)
    cbh.set_question("Q1: wny_health rows?", render=False)
    cbh.normal_message_flow("Q1: wny_health rows?")

    # Second question
    monkeypatch.setattr(cbh, "get_vn", lambda: _DummyVNService(second_thinking))
    cbh.set_question("Q2: wny_health rows again?", render=False)
    cbh.normal_message_flow("Q2: wny_health rows again?")

    full_seq = [m.type for m in fake_st.session_state.messages]
    # Find last two user messages
    user_positions = [i for i, t in enumerate(full_seq) if t == "text"]
    assert len(user_positions) >= 2, f"Expected at least 2 user text messages, got {user_positions} seq={full_seq}"
    last_pos = user_positions[-1]

    if second_thinking:
        must_prefix = ["text", "thinking", "sql", "dataframe", "summary"]
    else:
        must_prefix = ["text", "sql", "dataframe", "summary"]

    assert full_seq[last_pos:last_pos+len(must_prefix)] == must_prefix, (
        f"Second question sequence wrong. Got {full_seq[last_pos:last_pos+len(must_prefix)]}, full={full_seq}"
    )
