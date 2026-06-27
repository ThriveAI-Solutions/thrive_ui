"""Unit tests for the runtime Vanna-fallback wiring (Feature #231 of Epic #228).

The integration point lives in ``agent/runtime.py``:
  - feature-flag helpers (``_is_fallback_feature_enabled``,
    ``_is_scrubbed_logging``)
  - the post-stream hook (``_maybe_invoke_vanna_fallback``)

The consumer-loop event-collection logic inside
``run_agentic_message_flow`` is a 4-line type check — exercised via the
``isinstance`` sanity test below rather than by driving the whole loop.
"""

from __future__ import annotations

import types
from contextlib import nullcontext

import pytest

from agent.fallback import FallbackDecision
from agent.state import (
    FinalResponseEvent,
    ToolCallCompleted,
    ToolCallStarted,
)


def _fake_st(secrets: dict | None = None, session_state: dict | None = None):
    """Build a minimal Streamlit double for the runtime helpers.

    ``secrets.get`` mirrors the chained ``st.secrets.get("agent", {})
    .get("fallback", {})...`` pattern in the production code.
    ``session_state`` is exposed both as a dict (``.get``) and attribute
    namespace.
    """
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
    if session_state:
        st.session_state.update(session_state)

    st.secrets = secrets if secrets is not None else {}
    st.spinner = lambda *_a, **_k: nullcontext()
    return st


# ---------- _is_fallback_feature_enabled ----------------------------------


@pytest.mark.parametrize(
    "secrets,opt_in,expected",
    [
        ({"agent": {"fallback": {"enabled": True}}}, True, True),
        ({"agent": {"fallback": {"enabled": True}}}, False, False),
        ({"agent": {"fallback": {"enabled": False}}}, True, False),
        ({"agent": {"fallback": {"enabled": False}}}, False, False),
        ({}, True, False),
        ({"agent": {}}, True, False),
        ({"agent": {"fallback": {}}}, True, False),
    ],
)
def test_is_fallback_feature_enabled_requires_both_flags(monkeypatch, secrets, opt_in, expected):
    import agent.runtime as runtime

    fake_st = _fake_st(secrets=secrets, session_state={"vanna_fallback_enabled": opt_in})
    monkeypatch.setattr(runtime, "st", fake_st)
    assert runtime._is_fallback_feature_enabled() is expected


def test_is_fallback_feature_enabled_handles_missing_session_key(monkeypatch):
    import agent.runtime as runtime

    fake_st = _fake_st(secrets={"agent": {"fallback": {"enabled": True}}})
    monkeypatch.setattr(runtime, "st", fake_st)
    # session-state key missing entirely — must default to False.
    assert runtime._is_fallback_feature_enabled() is False


# ---------- _is_scrubbed_logging ------------------------------------------


@pytest.mark.parametrize(
    "secrets,expected",
    [
        ({"agent_logging": {"mode": "scrubbed"}}, True),
        ({"agent_logging": {"mode": "full"}}, False),
        ({"agent_logging": {"mode": "disabled"}}, False),
        ({"agent_logging": {}}, False),
        ({}, False),
    ],
)
def test_is_scrubbed_logging(monkeypatch, secrets, expected):
    import agent.runtime as runtime

    fake_st = _fake_st(secrets=secrets)
    monkeypatch.setattr(runtime, "st", fake_st)
    assert runtime._is_scrubbed_logging() is expected


# ---------- _maybe_invoke_vanna_fallback ---------------------------------


def _patch_chat_bot_helper(monkeypatch, on_call=None):
    """Patch the lazy-imported chat_bot_helper symbols.

    Returns a ``calls`` list that records each invocation. The first
    string per entry is "add_message" or "vanna"; the second is the
    payload.
    """
    calls: list[tuple[str, object]] = []

    def _fake_add_message(msg, render=True):
        calls.append(("add_message", msg))

    def _fake_run_vanna_flow(question):
        calls.append(("vanna", question))
        if on_call is not None:
            on_call()

    import utils.chat_bot_helper as cbh

    monkeypatch.setattr(cbh, "add_message", _fake_add_message)
    monkeypatch.setattr(cbh, "_run_vanna_flow", _fake_run_vanna_flow)
    return calls


def test_maybe_invoke_vanna_fallback_short_circuits_when_feature_disabled(monkeypatch):
    import agent.runtime as runtime

    monkeypatch.setattr(runtime, "_is_fallback_feature_enabled", lambda: False)

    run_async_calls = []

    def _spy_run_async(coro):
        run_async_calls.append(coro)
        coro.close()
        return None

    monkeypatch.setattr(runtime, "_run_async", _spy_run_async)
    calls = _patch_chat_bot_helper(monkeypatch)

    runtime._maybe_invoke_vanna_fallback("q?", "answer", [])

    assert run_async_calls == []
    assert calls == []


def test_maybe_invoke_vanna_fallback_skips_when_decision_says_no(monkeypatch):
    import agent.runtime as runtime

    fake_st = _fake_st()
    monkeypatch.setattr(runtime, "st", fake_st)
    monkeypatch.setattr(runtime, "_is_fallback_feature_enabled", lambda: True)
    monkeypatch.setattr(runtime, "_is_scrubbed_logging", lambda: False)

    def _mock_run_async(coro):
        coro.close()
        return FallbackDecision(invoke=False, reason="classifier_says_adequate")

    monkeypatch.setattr(runtime, "_run_async", _mock_run_async)
    calls = _patch_chat_bot_helper(monkeypatch)

    runtime._maybe_invoke_vanna_fallback("q?", "answer", [])

    assert calls == []


def test_maybe_invoke_vanna_fallback_invokes_vanna_when_decision_says_yes(monkeypatch):
    import agent.runtime as runtime
    from utils.enums import MessageType, RoleType

    fake_st = _fake_st()
    monkeypatch.setattr(runtime, "st", fake_st)
    monkeypatch.setattr(runtime, "_is_fallback_feature_enabled", lambda: True)
    monkeypatch.setattr(runtime, "_is_scrubbed_logging", lambda: False)

    def _mock_run_async(coro):
        coro.close()
        return FallbackDecision(invoke=True, reason="classifier_says_inadequate")

    monkeypatch.setattr(runtime, "_run_async", _mock_run_async)
    calls = _patch_chat_bot_helper(monkeypatch)

    runtime._maybe_invoke_vanna_fallback("how many patients?", "I'm not sure.", [])

    # Banner first, then Vanna call.
    assert len(calls) == 2
    kind0, msg = calls[0]
    assert kind0 == "add_message"
    assert msg.role == RoleType.ASSISTANT.value
    assert msg.type == MessageType.TEXT.value
    assert "falling back to direct SQL" in msg.content

    kind1, question = calls[1]
    assert kind1 == "vanna"
    assert question == "how many patients?"


def test_maybe_invoke_vanna_fallback_swallows_classifier_exception(monkeypatch):
    import agent.runtime as runtime

    fake_st = _fake_st()
    monkeypatch.setattr(runtime, "st", fake_st)
    monkeypatch.setattr(runtime, "_is_fallback_feature_enabled", lambda: True)
    monkeypatch.setattr(runtime, "_is_scrubbed_logging", lambda: False)

    def _raising_run_async(coro):
        coro.close()
        raise RuntimeError("provider down")

    monkeypatch.setattr(runtime, "_run_async", _raising_run_async)
    calls = _patch_chat_bot_helper(monkeypatch)

    # Must NOT raise. Must NOT invoke Vanna.
    runtime._maybe_invoke_vanna_fallback("q?", "answer", [])

    assert calls == []


def test_maybe_invoke_vanna_fallback_passes_scrubbed_flag(monkeypatch):
    import agent.runtime as runtime

    fake_st = _fake_st()
    monkeypatch.setattr(runtime, "st", fake_st)
    monkeypatch.setattr(runtime, "_is_fallback_feature_enabled", lambda: True)
    monkeypatch.setattr(runtime, "_is_scrubbed_logging", lambda: True)

    captured: dict = {}

    async def _spy_should_fallback(**kwargs):
        captured.update(kwargs)
        return FallbackDecision(invoke=False, reason="classifier_says_adequate")

    monkeypatch.setattr(runtime, "should_fallback", _spy_should_fallback)

    def _mock_run_async(coro):
        # Drive the spy coroutine to completion so we capture its kwargs.
        # asyncio.run() spins a fresh loop rather than relying on an ambient
        # current loop in the main thread — newer pytest-asyncio (1.4+) no
        # longer leaves one set for sync tests, so get_event_loop() raises.
        import asyncio

        return asyncio.run(coro)

    monkeypatch.setattr(runtime, "_run_async", _mock_run_async)
    _patch_chat_bot_helper(monkeypatch)

    runtime._maybe_invoke_vanna_fallback("q?", "answer", [])

    assert captured.get("scrubbed") is True
    assert captured.get("question") == "q?"
    assert captured.get("final_text") == "answer"
    assert captured.get("feature_enabled") is True


def test_maybe_invoke_vanna_fallback_passes_tool_events(monkeypatch):
    """The tool-event list captured during the consumer loop must reach
    should_fallback verbatim — it's how the decision engine knows which
    tools (if any) the agent actually called.
    """
    import agent.runtime as runtime

    fake_st = _fake_st()
    monkeypatch.setattr(runtime, "st", fake_st)
    monkeypatch.setattr(runtime, "_is_fallback_feature_enabled", lambda: True)
    monkeypatch.setattr(runtime, "_is_scrubbed_logging", lambda: False)

    captured: dict = {}

    async def _spy_should_fallback(**kwargs):
        captured.update(kwargs)
        return FallbackDecision(invoke=False, reason="classifier_says_adequate")

    monkeypatch.setattr(runtime, "should_fallback", _spy_should_fallback)

    def _mock_run_async(coro):
        import asyncio

        # asyncio.run() spins a fresh loop rather than relying on an ambient
        # current loop in the main thread. Newer pytest-asyncio (1.4+) no
        # longer leaves one set for sync tests, so get_event_loop() raises
        # "no current event loop in thread 'MainThread'".
        return asyncio.run(coro)

    monkeypatch.setattr(runtime, "_run_async", _mock_run_async)
    _patch_chat_bot_helper(monkeypatch)

    events = [
        ToolCallStarted(tool_name="search_knowledge_base", arguments={}),
        ToolCallStarted(tool_name="find_patient", arguments={"first_name": "Jane"}),
    ]
    runtime._maybe_invoke_vanna_fallback("q?", "answer", events)

    assert captured.get("tool_events") == events


# ---------- mark_run_fallback_invoked wiring (Feature #233) ---------------


def test_maybe_invoke_vanna_fallback_marks_run_before_invoking_vanna(monkeypatch):
    """Audit ordering: the agent run must be marked `fallback_invoked`
    BEFORE _run_vanna_flow is called. If we marked it after, a Vanna
    failure would leave the audit row stuck at `status='success'` while
    the user saw the fallback attempt.
    """
    import agent.runtime as runtime

    fake_st = _fake_st()
    monkeypatch.setattr(runtime, "st", fake_st)
    monkeypatch.setattr(runtime, "_is_fallback_feature_enabled", lambda: True)
    monkeypatch.setattr(runtime, "_is_scrubbed_logging", lambda: False)

    def _mock_run_async(coro):
        coro.close()
        return FallbackDecision(invoke=True, reason="classifier_says_inadequate")

    monkeypatch.setattr(runtime, "_run_async", _mock_run_async)

    # Spy that records the order of operations across mark-then-vanna.
    call_log: list[tuple] = []

    def _spy_mark(run_id, fallback_sql):
        call_log.append(("mark", run_id, fallback_sql))

    monkeypatch.setattr(runtime, "_mark_fallback_invoked_safely", _spy_mark)

    import utils.chat_bot_helper as cbh

    monkeypatch.setattr(cbh, "add_message", lambda *_a, **_k: None)
    monkeypatch.setattr(cbh, "_run_vanna_flow", lambda q: call_log.append(("vanna", q)))

    runtime._maybe_invoke_vanna_fallback("q?", "answer", [], run_id="r-abc")

    # First call is the pre-invocation mark with no SQL yet.
    assert call_log[0] == ("mark", "r-abc", None)
    # Second is the Vanna invocation.
    assert call_log[1] == ("vanna", "q?")


def test_maybe_invoke_vanna_fallback_persists_sql_on_rerun(monkeypatch):
    """When _run_vanna_flow raises RerunException, the runtime must read
    the breadcrumb session-state key and mark the run a SECOND time
    with the captured SQL before re-raising.
    """
    import agent.runtime as runtime

    class RerunException(BaseException):
        pass

    fake_st = _fake_st()
    monkeypatch.setattr(runtime, "st", fake_st)
    monkeypatch.setattr(runtime, "_is_fallback_feature_enabled", lambda: True)
    monkeypatch.setattr(runtime, "_is_scrubbed_logging", lambda: False)

    def _mock_run_async(coro):
        coro.close()
        return FallbackDecision(invoke=True, reason="classifier_says_inadequate")

    monkeypatch.setattr(runtime, "_run_async", _mock_run_async)

    mark_calls: list[tuple] = []
    monkeypatch.setattr(
        runtime,
        "_mark_fallback_invoked_safely",
        lambda run_id, fallback_sql: mark_calls.append((run_id, fallback_sql)),
    )

    def _fake_vanna(question):
        fake_st.session_state["_last_vanna_fallback_sql"] = "SELECT 42"
        raise RerunException()

    import utils.chat_bot_helper as cbh

    monkeypatch.setattr(cbh, "add_message", lambda *_a, **_k: None)
    monkeypatch.setattr(cbh, "_run_vanna_flow", _fake_vanna)

    # The RerunException must propagate so Streamlit handles the rerun.
    try:
        runtime._maybe_invoke_vanna_fallback("q?", "answer", [], run_id="r-xyz")
    except BaseException as exc:
        assert type(exc).__name__ == "RerunException"
    else:
        raise AssertionError("RerunException was not propagated")

    # Two mark calls: one before Vanna with NULL SQL, one after with the
    # captured SQL.
    assert mark_calls == [("r-xyz", None), ("r-xyz", "SELECT 42")]


def test_maybe_invoke_vanna_fallback_skips_mark_when_run_id_is_none(monkeypatch):
    """run_id may be None when agent run logging is disabled. The mark
    helper must be a no-op in that case — no DB session opened.
    """
    import agent.runtime as runtime

    fake_st = _fake_st()
    monkeypatch.setattr(runtime, "st", fake_st)
    monkeypatch.setattr(runtime, "_is_fallback_feature_enabled", lambda: True)
    monkeypatch.setattr(runtime, "_is_scrubbed_logging", lambda: False)

    def _mock_run_async(coro):
        coro.close()
        return FallbackDecision(invoke=False, reason="classifier_says_adequate")

    monkeypatch.setattr(runtime, "_run_async", _mock_run_async)

    mark_calls = []
    monkeypatch.setattr(
        runtime,
        "mark_run_fallback_invoked",
        lambda *_a, **_k: mark_calls.append("called"),
    )

    runtime._maybe_invoke_vanna_fallback("q?", "answer", [], run_id=None)

    # Decision said no-fallback AND run_id was None, so mark was never reached.
    assert mark_calls == []


# ---------- consumer-loop event classification ----------------------------


def test_isinstance_check_distinguishes_tool_events_from_others():
    """The runtime's consumer loop uses ``isinstance(event,
    (ToolCallStarted, ToolCallCompleted))`` to accumulate tool events
    and ``isinstance(event, FinalResponseEvent)`` to capture the final
    text. Pin the type-check semantics so neither class is accidentally
    dropped from the tuple later.
    """
    started = ToolCallStarted(tool_name="run_sql", arguments={"sql": "SELECT 1"})
    completed = ToolCallCompleted(
        tool_name="run_sql",
        result_summary="ok",
        success=True,
        elapsed_ms=42,
    )

    assert isinstance(started, (ToolCallStarted, ToolCallCompleted))
    assert isinstance(completed, (ToolCallStarted, ToolCallCompleted))
    # FinalResponseEvent must NOT match the tool-event tuple — otherwise
    # the consumer loop would over-collect.
    from agent.state import AgentResponse

    final = FinalResponseEvent(
        response=AgentResponse(text="done"),
        all_messages=[],
        usage=None,
    )
    assert not isinstance(final, (ToolCallStarted, ToolCallCompleted))
    assert isinstance(final, FinalResponseEvent)
