"""Pins the Vanna-flow dispatcher contract (Feature #229, Epic #228).

`_run_message_flow` is the seam between the agentic and legacy Vanna paths.
The Vanna body was extracted into `_run_vanna_flow` so it can be invoked
independently — both directly from the dispatcher and (in Feature #231) as
the agent's fallback path. These tests defend against accidental re-inlining
of the Vanna body back into the dispatcher.
"""

import inspect
import types
from contextlib import nullcontext


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
    st.chat_message = lambda *_a, **_k: nullcontext()
    st.empty = lambda: None
    st.rerun = lambda: None
    st.toast = lambda *_a, **_k: None
    st.markdown = lambda *_a, **_k: None
    return st


def test_run_vanna_flow_is_a_callable_with_question_param():
    from utils.chat_bot_helper import _run_vanna_flow

    assert callable(_run_vanna_flow)
    sig = inspect.signature(_run_vanna_flow)
    assert list(sig.parameters) == ["my_question"]


def test_run_message_flow_is_a_callable_with_question_param():
    from utils.chat_bot_helper import _run_message_flow

    assert callable(_run_message_flow)
    sig = inspect.signature(_run_message_flow)
    assert list(sig.parameters) == ["my_question"]


def test_dispatcher_routes_to_agent_when_agentic_mode_on(monkeypatch):
    import utils.chat_bot_helper as cbh

    fake_st = _fake_st()
    fake_st.session_state["agentic_mode"] = True
    monkeypatch.setattr(cbh, "st", fake_st)

    agent_calls = []

    def _fake_agent_flow(q):
        agent_calls.append(q)

    import agent.runtime as runtime_mod

    monkeypatch.setattr(runtime_mod, "run_agentic_message_flow", _fake_agent_flow)

    vanna_calls = []
    monkeypatch.setattr(cbh, "_run_vanna_flow", lambda q: vanna_calls.append(q))

    cbh._run_message_flow("how many patients?")

    assert agent_calls == ["how many patients?"]
    assert vanna_calls == []


def test_dispatcher_routes_to_vanna_when_agentic_mode_off(monkeypatch):
    import utils.chat_bot_helper as cbh

    fake_st = _fake_st()
    fake_st.session_state["agentic_mode"] = False
    monkeypatch.setattr(cbh, "st", fake_st)

    vanna_calls = []
    monkeypatch.setattr(cbh, "_run_vanna_flow", lambda q: vanna_calls.append(q))

    import agent.runtime as runtime_mod

    agent_calls = []
    monkeypatch.setattr(runtime_mod, "run_agentic_message_flow", lambda q: agent_calls.append(q))

    cbh._run_message_flow("how many patients?")

    assert vanna_calls == ["how many patients?"]
    assert agent_calls == []
