import pytest
from pydantic import ValidationError
from agent.state import (
    StreamEvent,
    ToolCallStarted,
    ToolCallCompleted,
    FinalResponseEvent,
    AgentResponse,
)
from agent.artifacts import Artifact


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


def test_tool_call_completed_carries_reliability_note():
    from agent.state import ToolCallCompleted

    evt = ToolCallCompleted(
        tool_name="get_patient_clinical_data",
        result_summary="row_count=4",
        success=True,
        elapsed_ms=120,
        reliability_note="LOINC coverage ~50%",
    )
    assert evt.reliability_note == "LOINC coverage ~50%"


def test_agent_response_artifacts_accepts_union_variants():
    from agent.state import AgentResponse
    from agent.artifacts import (
        ChartArtifact,
        DataFrameArtifact,
        SqlArtifact,
        SummaryArtifact,
    )

    response = AgentResponse(
        text="ok",
        artifacts=[
            SqlArtifact(sql="SELECT 1"),
            DataFrameArtifact(columns=["a"], rows=[[1]], row_count=1),
            ChartArtifact(plotly_json="{}"),
            SummaryArtifact(text="ok"),
        ],
    )
    assert len(response.artifacts) == 4
    assert response.artifacts[0].kind == "sql"
    assert response.artifacts[1].kind == "dataframe"
    assert response.artifacts[2].kind == "chart"
    assert response.artifacts[3].kind == "summary"


def test_agent_response_artifacts_rejects_unknown_variant():
    import pytest
    from pydantic import ValidationError
    from agent.state import AgentResponse

    with pytest.raises(ValidationError):
        AgentResponse(text="ok", artifacts=[{"kind": "spaghetti"}])
