"""Source-neutral case execution: turn history threading, judge integration,
and fail-closed patient resolution."""

import asyncio
from datetime import datetime

import pytest

from agent.deps import SelectedPatient
from agent.state import AgentResponse, FinalResponseEvent, ToolCallCompleted, ToolCallStarted
from evals.cases import NormalizedCase, NormalizedTurn
from evals.executor import CaseExecution, EvaluationResources, execute_case, execute_cases


class _FakeAdapter:
    def __init__(self):
        self.pop_calls = 0

    def pop_sql_log(self):
        self.pop_calls += 1
        return []


class _FakeRunner:
    """Yields one fixed event sequence per call, keyed by call order."""

    def __init__(self, event_sequences):
        self._event_sequences = event_sequences
        self.calls: list[dict] = []

    async def stream(self, prompt, deps=None, message_history=None):
        index = len(self.calls)
        self.calls.append({"prompt": prompt, "message_history": message_history})
        for evt in self._event_sequences[index]:
            yield evt


def _events_for(answer: str, all_messages: list):
    return [
        ToolCallStarted(tool_name="run_sql", arguments={}),
        ToolCallCompleted(tool_name="run_sql", result_summary="1 row", success=True, elapsed_ms=10, sql_executed=[]),
        FinalResponseEvent(response=AgentResponse(text=answer), all_messages=all_messages, usage=None),
    ]


def _two_turn_case() -> NormalizedCase:
    return NormalizedCase(
        case_id="case-1",
        version=1,
        source_type="feedback",
        title="t",
        patient_source_id="src-1",
        patient_label="Jane Doe (label)",
        turns=(
            NormalizedTurn(role="main", prompt="Does this patient have diabetes?"),
            NormalizedTurn(role="followup", prompt="What is the most recent A1C?"),
        ),
        reviewer_guidance="Concerned the reported date was wrong",
    )


def _selected_patient(source_id: str) -> SelectedPatient:
    return SelectedPatient(
        source_id=source_id,
        display_name="Jane Doe",
        dob=None,
        selected_at=datetime.now(),
        selection_origin="user_click",
    )


@pytest.fixture
def fake_adapter():
    return _FakeAdapter()


def test_followup_turn_receives_prior_all_messages(monkeypatch, fake_adapter):
    monkeypatch.setattr("evals.executor.resolve_patient", lambda adapter, source_id: _selected_patient(source_id))
    runner = _FakeRunner(
        [
            _events_for("Yes.", all_messages=["m1", "m2"]),
            _events_for("7.2", all_messages=["m1", "m2", "m3"]),
        ]
    )
    resources = EvaluationResources(runner=runner, adapter=fake_adapter, rag=None, judge=None)
    case = _two_turn_case()

    result = asyncio.run(execute_case(case, resources))

    assert result.status == "completed"
    assert len(runner.calls) == 2
    assert runner.calls[0]["message_history"] is None
    assert runner.calls[1]["message_history"] == ["m1", "m2"]
    assert fake_adapter.pop_calls == 1


def test_tool_evidence_and_latency_survive_normalization(monkeypatch, fake_adapter):
    monkeypatch.setattr("evals.executor.resolve_patient", lambda adapter, source_id: _selected_patient(source_id))
    runner = _FakeRunner([_events_for("Yes.", all_messages=[])])
    resources = EvaluationResources(runner=runner, adapter=fake_adapter, rag=None, judge=None)
    case = NormalizedCase(
        case_id="case-1",
        version=1,
        source_type="curated",
        title="t",
        patient_source_id="src-1",
        patient_label="",
        turns=(NormalizedTurn(role="main", prompt="Q?"),),
    )

    result = asyncio.run(execute_case(case, resources))

    (turn,) = result.turns
    (tool_call,) = turn["tool_calls"]
    assert tool_call["tool_name"] == "run_sql"
    assert tool_call["success"] is True
    assert "latency" in turn
    assert turn["total_elapsed_ms"] >= 0


def test_judge_receives_original_feedback_concern(monkeypatch, fake_adapter):
    monkeypatch.setattr("evals.executor.resolve_patient", lambda adapter, source_id: _selected_patient(source_id))
    captured = []

    async def _fake_judge_turn(judge, prompt, answer, summaries, reviewer_guidance=None):
        captured.append(reviewer_guidance)
        return {"suggestion": "looks_correct", "reason": "consistent"}

    monkeypatch.setattr("evals.executor.judge_turn", _fake_judge_turn)
    runner = _FakeRunner([_events_for("Yes.", all_messages=[])])
    resources = EvaluationResources(runner=runner, adapter=fake_adapter, rag=None, judge=object())
    case = NormalizedCase(
        case_id="case-1",
        version=1,
        source_type="feedback",
        title="t",
        patient_source_id="src-1",
        patient_label="",
        turns=(NormalizedTurn(role="main", prompt="Q?"),),
        reviewer_guidance="Concerned the reported date was wrong",
    )

    result = asyncio.run(execute_case(case, resources))

    assert result.turns[0]["judge"]["suggestion"] == "looks_correct"
    assert captured == ["Concerned the reported date was wrong"]


def test_judge_failure_yields_none(monkeypatch, fake_adapter):
    monkeypatch.setattr("evals.executor.resolve_patient", lambda adapter, source_id: _selected_patient(source_id))

    async def _failing_judge_turn(judge, prompt, answer, summaries, reviewer_guidance=None):
        return None

    monkeypatch.setattr("evals.executor.judge_turn", _failing_judge_turn)
    runner = _FakeRunner([_events_for("Yes.", all_messages=[])])
    resources = EvaluationResources(runner=runner, adapter=fake_adapter, rag=None, judge=object())
    case = NormalizedCase(
        case_id="case-1",
        version=1,
        source_type="curated",
        title="t",
        patient_source_id="src-1",
        patient_label="",
        turns=(NormalizedTurn(role="main", prompt="Q?"),),
    )

    result = asyncio.run(execute_case(case, resources))

    assert result.turns[0]["judge"] is None


def test_patient_resolution_failure_fails_closed(monkeypatch, fake_adapter):
    def _raise(adapter, source_id):
        raise LookupError(f"source_id {source_id!r} not found in warehouse")

    monkeypatch.setattr("evals.executor.resolve_patient", _raise)
    runner = _FakeRunner([])
    resources = EvaluationResources(runner=runner, adapter=fake_adapter, rag=None, judge=None)
    case = NormalizedCase(
        case_id="case-1",
        version=1,
        source_type="curated",
        title="t",
        patient_source_id="src-missing",
        patient_label="",
        turns=(NormalizedTurn(role="main", prompt="Q?"),),
    )

    result = asyncio.run(execute_case(case, resources))

    assert result.status == "failed"
    assert result.error_type == "LookupError"
    assert result.patient["source_id"] == "src-missing"
    assert result.patient["display_name"] == ""
    assert result.turns == ()
    assert runner.calls == []


def test_execute_cases_invokes_callback_after_every_case(monkeypatch, fake_adapter):
    monkeypatch.setattr("evals.executor.resolve_patient", lambda adapter, source_id: _selected_patient(source_id))
    runner = _FakeRunner(
        [
            _events_for("A1.", all_messages=[]),
            _events_for("A2.", all_messages=[]),
        ]
    )
    resources = EvaluationResources(runner=runner, adapter=fake_adapter, rag=None, judge=None)
    cases = [
        NormalizedCase(
            case_id=cid,
            version=1,
            source_type="curated",
            title="t",
            patient_source_id="src-1",
            patient_label="",
            turns=(NormalizedTurn(role="main", prompt="Q?"),),
        )
        for cid in ("case-1", "case-2")
    ]
    completed: list[CaseExecution] = []

    results = asyncio.run(execute_cases(cases, resources, completed.append))

    assert [c.case_id for c in completed] == ["case-1", "case-2"]
    assert results == completed
