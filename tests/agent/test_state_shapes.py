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


from agent.tools.get_patient_clinical_data import (
    SurgeriesQuery,
    SurgeryItem,
    PatientClinicalQuery,
    ClinicalItem,
)
from pydantic import TypeAdapter


def test_surgeries_query_in_discriminated_union():
    adapter = TypeAdapter(PatientClinicalQuery)
    q = adapter.validate_python({"domain": "surgeries"})
    assert isinstance(q, SurgeriesQuery)


def test_surgeries_query_with_filters():
    adapter = TypeAdapter(PatientClinicalQuery)
    q = adapter.validate_python({
        "domain": "surgeries",
        "cpt_codes": ["27447"],
        "procedure_text": "knee",
        "date_range": {"start": "2025-01-01"},
    })
    assert isinstance(q, SurgeriesQuery)
    assert q.cpt_codes == ["27447"]


def test_surgery_item_in_clinical_item_union():
    adapter = TypeAdapter(ClinicalItem)
    item = adapter.validate_python({
        "item_type": "surgery",
        "source": "orders",
        "source_id": "src-1",
        "code": "27447",
        "code_type": "CPT",
        "description": "Total knee arthroplasty",
        "event_date": "2025-06-15",
        "place_of_service": "21",
        "provider_npi": None,
        "performing_provider": "Dr. Ortho",
        "provider_ambiguous": False,
        "facility_name": None,
    })
    assert isinstance(item, SurgeryItem)
    assert item.performing_provider == "Dr. Ortho"
    assert item.provider_ambiguous is False
