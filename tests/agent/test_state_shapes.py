import pytest
from pydantic import ValidationError
from agent.state import (
    StreamEvent,
    ToolCallStarted,
    ToolCallCompleted,
    FinalResponseEvent,
    AgentResponse,
    Artifact,
)


def test_tool_call_started_event():
    e = ToolCallStarted(tool_name="find_patient", arguments={"first_name": "John"})
    assert e.kind == "tool_call_started"


def test_tool_call_completed_event():
    e = ToolCallCompleted(
        tool_name="find_patient",
        result_summary="match_count=5",
        success=True,
        elapsed_ms=120,
    )
    assert e.kind == "tool_call_completed"
    assert e.success is True


def test_agent_response_minimal():
    r = AgentResponse(text="Found the patient.")
    assert r.followups == []
    assert r.clear_selection is False


def test_agent_response_with_clear_selection():
    r = AgentResponse(text="Stepping back.", clear_selection=True)
    assert r.clear_selection is True


def test_agent_response_with_followups():
    r = AgentResponse(
        text="Done.",
        followups=["What about labs?", "Show medications"],
    )
    assert len(r.followups) == 2
