"""runtime.run_agentic_message_flow must:

1. Stash the original question as `pending_user_question` so the chooser
   click handler can re-trigger the flow with it.
2. Pass `agent_message_history` from session_state to runner.stream() so
   prior turns continue the same conversation.
3. After the run completes, persist `run.result.all_messages()` back to
   session_state under `agent_message_history` for the next turn.
4. Clear `my_question` after the run (parity with the Vanna flow).
5. Wipe `agent_message_history` and selection keys when the agent
   returns final_result with clear_selection=True.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent.state import AgentResponse, FinalResponseEvent


class _FakeSession:
    def __init__(self):
        self.committed = False
        self.closed = False

    def add(self, *a, **kw):
        pass

    def flush(self):
        pass

    def commit(self):
        self.committed = True

    def close(self):
        self.closed = True


def _runner_yielding(events):
    """A runner whose stream() yields the given events in order. Records
    the message_history kwarg it was called with for assertions."""

    captured = {"message_history": "<unset>"}

    class _Runner:
        async def stream(self, question, deps, message_history=None):
            captured["message_history"] = message_history
            for ev in events:
                yield ev

    return _Runner(), captured


def _final_event(text="ok", clear_selection=False, all_messages=None):
    return FinalResponseEvent(
        response=AgentResponse(
            text=text,
            followups=[],
            artifacts=[],
            clear_selection=clear_selection,
        ),
        all_messages=all_messages or ["msg-a", "msg-b"],
    )


def _patch_runtime(monkeypatch, fake_state, runner):
    import agent.runtime as runtime_mod

    monkeypatch.setattr(runtime_mod, "SessionLocal", lambda: _FakeSession())
    monkeypatch.setattr(runtime_mod, "build_agent_deps", lambda s: MagicMock())
    monkeypatch.setattr(runtime_mod, "_runner", lambda: runner)
    monkeypatch.setattr("utils.chat_bot_helper.add_message", lambda *a, **kw: None, raising=False)

    class _StSeam:
        session_state = fake_state

    monkeypatch.setattr(runtime_mod, "st", _StSeam, raising=True)
    return runtime_mod


def test_pending_user_question_is_stashed_before_run(monkeypatch):
    fake_state = {"my_question": "Has John Smith had any procedures?"}
    runner, _ = _runner_yielding([_final_event()])
    runtime_mod = _patch_runtime(monkeypatch, fake_state, runner)

    runtime_mod.run_agentic_message_flow("Has John Smith had any procedures?")

    assert fake_state["pending_user_question"] == "Has John Smith had any procedures?"


def test_my_question_is_cleared_after_run(monkeypatch):
    fake_state = {"my_question": "anything"}
    runner, _ = _runner_yielding([_final_event()])
    runtime_mod = _patch_runtime(monkeypatch, fake_state, runner)

    runtime_mod.run_agentic_message_flow("anything")

    assert fake_state.get("my_question") is None


def test_message_history_is_persisted_to_session_state(monkeypatch):
    fake_state = {}
    runner, _ = _runner_yielding([_final_event(all_messages=["m1", "m2", "m3"])])
    runtime_mod = _patch_runtime(monkeypatch, fake_state, runner)

    runtime_mod.run_agentic_message_flow("first turn")

    assert fake_state["agent_message_history"] == ["m1", "m2", "m3"]


def test_prior_message_history_is_passed_to_runner(monkeypatch):
    fake_state = {"agent_message_history": ["prior-1", "prior-2"]}
    runner, captured = _runner_yielding([_final_event()])
    runtime_mod = _patch_runtime(monkeypatch, fake_state, runner)

    runtime_mod.run_agentic_message_flow("follow-up question")

    assert captured["message_history"] == ["prior-1", "prior-2"]


def test_my_question_is_cleared_even_if_add_message_raises(monkeypatch):
    """If add_message fails (SQLite write error, session_state in an
    unexpected shape, etc.), my_question still has to be cleared.
    Otherwise the chat_bot.py loop will re-fire the same question on
    the next rerun and the user is stuck in a silent loop."""
    fake_state = {"my_question": "anything"}
    runner, _ = _runner_yielding([_final_event()])
    runtime_mod = _patch_runtime(monkeypatch, fake_state, runner)

    def boom(*a, **kw):
        raise RuntimeError("simulated message persistence failure")

    monkeypatch.setattr("utils.chat_bot_helper.add_message", boom, raising=False)

    with pytest.raises(RuntimeError, match="simulated message persistence failure"):
        runtime_mod.run_agentic_message_flow("anything")

    assert fake_state.get("my_question") is None


def test_clear_selection_wipes_message_history_and_selection_keys(monkeypatch):
    fake_state = {
        "agent_message_history": ["old-msg"],
        "selected_patient_source_id": "src-x",
        "selected_patient_display_name": "X",
        "selected_patient_dob": "1990-01-01",
        "selection_origin": "user_click",
        "selected_at": "2026-05-06",
    }
    runner, _ = _runner_yielding([_final_event(clear_selection=True)])
    runtime_mod = _patch_runtime(monkeypatch, fake_state, runner)

    runtime_mod.run_agentic_message_flow("start over")

    assert "agent_message_history" not in fake_state
    assert "selected_patient_source_id" not in fake_state
    assert "selected_patient_display_name" not in fake_state
    assert "selection_origin" not in fake_state
