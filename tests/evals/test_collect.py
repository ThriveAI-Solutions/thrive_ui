import asyncio

from agent.state import (
    AgentResponse,
    CapReachedEvent,
    FinalResponseEvent,
    ThinkingCompletedEvent,
    ToolCallCompleted,
    ToolCallStarted,
)
from evals.collect import run_turn


class _StubRunner:
    def __init__(self, events):
        self._events = events

    async def stream(self, prompt, deps=None, message_history=None):
        for e in self._events:
            yield e


def _events():
    return [
        ThinkingCompletedEvent(text="let me check procedures", elapsed_ms=400, turn_index=0),
        ToolCallStarted(tool_name="get_patient_clinical_data", arguments={"domain": "procedures"}),
        ToolCallCompleted(
            tool_name="get_patient_clinical_data",
            result_summary="3 rows; data_availability=data_present",
            success=True,
            elapsed_ms=2300,
            reliability_note="HEALTHeLINK feed only",
            sql_executed=[{"sql": "SELECT 1", "params": {}, "db_elapsed_ms": 2100}],
        ),
        FinalResponseEvent(
            response=AgentResponse(text="Yes — knee arthroplasty on 2025-06-15."),
            all_messages=["m1", "m2"],
            usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        ),
    ]


def test_collects_turn_fields():
    runner = _StubRunner(_events())
    turn, all_messages = asyncio.run(run_turn(runner, deps=None, prompt="Q?", message_history=None))
    assert turn["prompt"] == "Q?"
    assert turn["answer"].startswith("Yes")
    assert turn["thinking"] == ["let me check procedures"]
    assert turn["cap_reached"] is None
    assert turn["usage"]["total_tokens"] == 15
    assert all_messages == ["m1", "m2"]
    (tc,) = turn["tool_calls"]
    assert tc["tool_name"] == "get_patient_clinical_data"
    assert tc["arguments"] == {"domain": "procedures"}
    assert tc["success"] is True
    assert tc["sql_executed"][0]["db_elapsed_ms"] == 2100
    assert tc["reliability_note"] == "HEALTHeLINK feed only"
    assert turn["latency"]["redshift_ms"] == 2100
    assert turn["total_elapsed_ms"] >= 0


def test_records_cap_event():
    events = _events()[:3] + [CapReachedEvent(reason="tool_count")]
    turn, all_messages = asyncio.run(run_turn(_StubRunner(events), deps=None, prompt="Q?", message_history=None))
    assert turn["cap_reached"] == "tool_count"
    assert turn["answer"] == ""
    assert all_messages == []


def test_orphan_completed_creates_entry():
    events = [
        ToolCallCompleted(
            tool_name="run_sql",
            result_summary="1 row",
            success=True,
            elapsed_ms=50,
        ),
        FinalResponseEvent(response=AgentResponse(text="done"), all_messages=[], usage=None),
    ]
    turn, _ = asyncio.run(run_turn(_StubRunner(events), deps=None, prompt="Q?", message_history=None))
    (tc,) = turn["tool_calls"]
    assert tc["arguments"] == {}
    assert tc["completed"] is True
    assert tc["success"] is True
