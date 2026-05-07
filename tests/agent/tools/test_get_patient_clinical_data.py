from datetime import date, datetime
from unittest.mock import MagicMock
import pytest

from agent.deps import AgentDeps, SelectedPatient
from agent.db.analytics_adapter import AnalyticsDbAdapter
from agent.tools.get_patient_clinical_data import (
    get_patient_clinical_data,
    DemographicsQuery,
    EncountersQuery,
    DateRange,
    ClinicalResult,
)
from pydantic_ai import ModelRetry


def _deps(synthetic_db, selected: SelectedPatient | None) -> AgentDeps:
    return AgentDeps(
        user_id=1,
        user_role=MagicMock(value=1),
        session_id="s1",
        selected_patient=selected,
        last_dataframe=None,
        last_sql=None,
        last_query_meta=None,
        analytics_db=AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite"),
        rag=None,
        sqlite_session=None,
        audit_logger=MagicMock(),
    )


def _selected_john() -> SelectedPatient:
    return SelectedPatient(
        source_id="src-john-1962",
        display_name="John Smith",
        dob=date(1962, 5, 1),
        selected_at=datetime.now(),
        selection_origin="user_click",
    )


def test_demographics_returns_data_present(synthetic_db):
    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, _selected_john())
    result = get_patient_clinical_data(ctx, DemographicsQuery())
    assert isinstance(result, ClinicalResult)
    assert result.domain == "demographics"
    assert result.data_availability == "data_present"
    assert len(result.items) == 1


def test_encounters_returns_three_for_john_1962(synthetic_db):
    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, _selected_john())
    result = get_patient_clinical_data(ctx, EncountersQuery())
    assert result.domain == "encounters"
    assert len(result.items) == 3


def test_encounters_date_range_narrows_results(synthetic_db):
    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, _selected_john())
    q = EncountersQuery(date_range=DateRange(start=date(2026, 3, 1), end=date(2026, 4, 30)))
    result = get_patient_clinical_data(ctx, q)
    assert len(result.items) == 1


def test_no_selection_raises_model_retry(synthetic_db):
    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, selected=None)
    with pytest.raises(ModelRetry, match="No patient is currently selected"):
        get_patient_clinical_data(ctx, EncountersQuery())


def test_no_records_found_data_availability(synthetic_db):
    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, _selected_john())
    q = EncountersQuery(date_range=DateRange(start=date(1900, 1, 1), end=date(1900, 12, 31)))
    result = get_patient_clinical_data(ctx, q)
    assert result.data_availability == "no_records_found"
    assert result.items == []


from agent.tools.get_patient_clinical_data import LabsQuery, LabItem


def test_labs_returns_data_present(synthetic_db):
    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, _selected_john())
    result = get_patient_clinical_data(ctx, LabsQuery())
    assert result.domain == "labs"
    assert result.data_availability == "data_present"
    assert len(result.items) == 4
    assert all(isinstance(i, LabItem) for i in result.items)


def test_labs_reliability_note_set_when_non_loinc_present(synthetic_db):
    """At least one fixture row has code_type='local' — that triggers the badge."""
    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, _selected_john())
    result = get_patient_clinical_data(ctx, LabsQuery())
    assert result.reliability_note is not None
    assert "loinc" in result.reliability_note.lower()


def test_labs_negative_result_filter(synthetic_db):
    from agent.tools.get_patient_clinical_data import LabsQuery

    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, _selected_john())
    result = get_patient_clinical_data(ctx, LabsQuery(result_filter="negative"))
    assert result.data_availability == "data_present"
    assert len(result.items) == 1
    assert result.items[0].clean_result == "negative"


from agent.tools.get_patient_clinical_data import DiagnosesQuery, DiagnosisItem


def test_diagnoses_returns_four_for_john_1962(synthetic_db):
    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, _selected_john())
    result = get_patient_clinical_data(ctx, DiagnosesQuery())
    assert result.domain == "diagnoses"
    assert result.data_availability == "data_present"
    assert len(result.items) == 4
    assert all(isinstance(i, DiagnosisItem) for i in result.items)


def test_diagnoses_filtered_by_text_returns_diabetes(synthetic_db):
    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, _selected_john())
    q = DiagnosesQuery(condition_text="diabetes")
    result = get_patient_clinical_data(ctx, q)
    assert len(result.items) == 1
    assert "diabetes" in result.items[0].diagnosis.lower()


def test_diagnoses_most_recent_only(synthetic_db):
    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, _selected_john())
    q = DiagnosesQuery(most_recent_only=True)
    result = get_patient_clinical_data(ctx, q)
    assert len(result.items) == 1


from agent.tools.get_patient_clinical_data import MedicationsQuery, MedicationItem


def test_medications_returns_two(synthetic_db):
    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, _selected_john())
    result = get_patient_clinical_data(ctx, MedicationsQuery())
    assert result.domain == "medications"
    assert result.data_availability == "data_present"
    assert len(result.items) == 2
    assert all(isinstance(i, MedicationItem) for i in result.items)


def test_medications_filtered_by_rxnorm(synthetic_db):
    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, _selected_john())
    q = MedicationsQuery(rxnorm_codes=["18631"])
    result = get_patient_clinical_data(ctx, q)
    assert len(result.items) == 1
    assert result.items[0].med_name == "Azithromycin"


from agent.tools.get_patient_clinical_data import ImmunizationsQuery, ImmunizationItem


def test_immunizations_returns_two(synthetic_db):
    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, _selected_john())
    result = get_patient_clinical_data(ctx, ImmunizationsQuery())
    assert result.domain == "immunizations"
    assert result.data_availability == "data_present"
    assert len(result.items) == 2
    assert all(isinstance(i, ImmunizationItem) for i in result.items)


def test_immunizations_filtered_by_cvx(synthetic_db):
    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, _selected_john())
    q = ImmunizationsQuery(cvx_codes=["03"])
    result = get_patient_clinical_data(ctx, q)
    assert len(result.items) == 1
    assert "Measles" in result.items[0].vaccine


from agent.tools.get_patient_clinical_data import ProceduresQuery, ProcedureItem


def test_procedures_returns_three_sources(synthetic_db):
    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, _selected_john())
    result = get_patient_clinical_data(ctx, ProceduresQuery())
    assert result.domain == "procedures"
    assert result.data_availability == "data_present"
    sources = {i.source for i in result.items}
    assert sources == {"orders", "problems", "claims"}


def test_procedures_carries_claims_freshness_note(synthetic_db):
    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, _selected_john())
    result = get_patient_clinical_data(ctx, ProceduresQuery())
    assert result.reliability_note is not None
    assert "claims" in result.reliability_note.lower()


def test_procedures_filtered_by_cpt(synthetic_db):
    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, _selected_john())
    q = ProceduresQuery(cpt_codes=["45378"])
    result = get_patient_clinical_data(ctx, q)
    assert len(result.items) == 1


from agent.tools.get_patient_clinical_data import ImagingQuery, ImagingItem


def test_imaging_returns_data_present(synthetic_db):
    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, _selected_john())
    result = get_patient_clinical_data(ctx, ImagingQuery())
    assert result.domain == "imaging"
    assert result.data_availability == "data_present"
    assert len(result.items) >= 1


def test_imaging_always_carries_impression_unavailable_note(synthetic_db):
    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, _selected_john())
    result = get_patient_clinical_data(ctx, ImagingQuery())
    assert result.notes_to_agent is not None
    assert "impression" in result.notes_to_agent.lower()
