from datetime import date, datetime
from typing import get_args
from unittest.mock import MagicMock
import pytest

from agent.deps import AgentDeps, SelectedPatient
from agent.db.analytics_adapter import AnalyticsDbAdapter
from agent.tools.get_patient_clinical_data import (
    AdmissionStay,
    AllergyItem,
    ClinicalResult,
    DateRange,
    DemographicsItem,
    DemographicsQuery,
    DiagnosesQuery,
    DiagnosisItem,
    EncounterItem,
    EncountersQuery,
    ImagingItem,
    ImagingQuery,
    ImmunizationItem,
    ImmunizationsQuery,
    LabItem,
    LabsQuery,
    MedicationItem,
    MedicationsQuery,
    ProcedureItem,
    ProceduresQuery,
    SurgeriesQuery,
    SurgeryItem,
    _effective_med_date_stopped,
    _maybe_drug_allergy_signal,
    _normalized_problem_status,
    get_patient_clinical_data,
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
        run_logger=MagicMock(),
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


def test_encounters_returns_four_for_john_1962(synthetic_db):
    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, _selected_john())
    result = get_patient_clinical_data(ctx, EncountersQuery())
    assert result.domain == "encounters"
    assert len(result.items) == 4


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


def test_labs_returns_data_present(synthetic_db):
    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, _selected_john())
    result = get_patient_clinical_data(ctx, LabsQuery())
    assert result.domain == "labs"
    assert result.data_availability == "data_present"
    assert len(result.items) == 4
    assert all(isinstance(i, LabItem) for i in result.items)
    labs = [i for i in result.items if isinstance(i, LabItem)]
    assert {i.source_name for i in labs} == {"Buffalo Medical Group"}
    assert {i.service_provider for i in labs} == {"BMG Lab"}


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


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        ({"status": "TERMED"}, "inactive"),
        ({"status": "55561003"}, "active"),
        ({"status": "413322009"}, "resolved"),
        ({"status": "COMPLETED"}, "resolved"),
        ({"status": "Resolved"}, "resolved"),
        ({"status": "Working Dx"}, "Working Dx"),
        ({"status": "TERMED", "chronic_ind": "Y"}, "inactive"),
        ({"status": "  ", "chronic_ind": "N"}, None),
    ],
)
def test_problem_status_normalization_is_exact(row, expected):
    """chronic_ind is never consulted — a legacy flag, not a status."""
    assert _normalized_problem_status(row) == expected


def test_diagnoses_surface_normalized_status(synthetic_db):
    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, _selected_john())
    result = get_patient_clinical_data(ctx, DiagnosesQuery())
    by_code = {i.code: i for i in result.items if isinstance(i, DiagnosisItem)}
    # E11.9 has chronic_ind='Y' in the fixture but status '55561003' → "active";
    # the legacy chronic flag must not override the normalized SNOMED status.
    assert by_code["E11.9"].status == "active"
    assert by_code["B16.9"].status == "resolved"
    assert by_code["0DTJ4ZZ"].status == "resolved"


def test_medications_returns_two(synthetic_db):
    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, _selected_john())
    result = get_patient_clinical_data(ctx, MedicationsQuery())
    assert result.domain == "medications"
    assert result.data_availability == "data_present"
    assert len(result.items) == 2
    assert all(isinstance(i, MedicationItem) for i in result.items)


def test_medications_surface_status(synthetic_db):
    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, _selected_john())
    result = get_patient_clinical_data(ctx, MedicationsQuery())
    assert len(result.items) == 2
    by_name = {i.med_name: i for i in result.items if isinstance(i, MedicationItem)}
    assert by_name["Metformin"].status == "active"
    assert by_name["Metformin"].date_stopped is None
    assert by_name["Azithromycin"].status == "completed"
    assert by_name["Azithromycin"].date_stopped == "2026-04-07 00:00"


@pytest.mark.parametrize("status", ["Discontinued", "No Longer Active", "SUSPENDED", "On Hold"])
def test_inactive_medication_uses_status_date_as_stop_date(status):
    assert (
        _effective_med_date_stopped({"status": status, "status_date": "2024-06-01", "date_stopped": None})
        == "2024-06-01"
    )


def test_medication_explicit_stop_date_wins_and_active_status_date_is_not_stop():
    assert (
        _effective_med_date_stopped(
            {"status": "Discontinued", "status_date": "2024-06-01", "date_stopped": "2023-01-01"}
        )
        == "2023-01-01"
    )
    assert _effective_med_date_stopped({"status": "Active", "status_date": "2024-06-01"}) is None


def test_medications_blank_numeric_strings_coerce_to_none(synthetic_db):
    """The warehouse delivers drug_supply_days as VARCHAR with '' for missing
    (2026-07-06 prod incident: every meds call for ADT-feed patients died in
    MedicationItem validation, sending the agent into retry loops)."""
    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, _selected_john())
    result = get_patient_clinical_data(ctx, MedicationsQuery())
    by_name = {i.med_name: i for i in result.items}
    assert by_name["Azithromycin"].drug_supply_days is None
    assert by_name["Metformin"].drug_supply_days == 90


def test_medication_item_blank_and_numeric_string_coercion():
    assert MedicationItem(source_id="x", drug_supply_days="", number_of_refills=" ").drug_supply_days is None
    assert MedicationItem(source_id="x", number_of_refills="  ").number_of_refills is None
    assert MedicationItem(source_id="x", drug_supply_days="30").drug_supply_days == 30


_ITEM_MODELS = [
    DemographicsItem,
    EncounterItem,
    LabItem,
    DiagnosisItem,
    MedicationItem,
    ImmunizationItem,
    ProcedureItem,
    SurgeryItem,
    ImagingItem,
    AdmissionStay,
    AllergyItem,
]


@pytest.mark.parametrize("model", _ITEM_MODELS, ids=lambda m: m.__name__)
def test_item_models_tolerate_blank_strings(model):
    """Warehouse varchar columns deliver '' (not NULL) for missing values, and
    item fields are raw column passthroughs. Any str/int/date field must accept
    '' without a ValidationError (2026-07-06: drug_supply_days='' killed every
    meds retrieval). bool and Literal fields are computed/normalized in
    Python or SQL, never fed raw varchar, so they're exempt."""
    kwargs = {}
    for fname, f in model.model_fields.items():
        ann = str(f.annotation)
        if "Literal" in ann:
            if f.is_required():
                kwargs[fname] = get_args(f.annotation)[0]
            continue
        if "bool" in ann:
            continue
        kwargs[fname] = "x" if f.is_required() else ""
    model(**kwargs)


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


def test_procedures_returns_orders_and_problems(synthetic_db):
    """The claims branch is hard-suppressed in Phase 3 (no patient ID on
    federated_claims_icd_procedure_detail_v); only orders + problems
    surface."""
    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, _selected_john())
    result = get_patient_clinical_data(ctx, ProceduresQuery())
    assert result.domain == "procedures"
    assert result.data_availability == "data_present"
    sources = {i.source for i in result.items}
    assert sources == {"orders", "problems"}
    assert "claims" not in sources


def test_procedures_reliability_note_flags_claims_suppression(synthetic_db):
    """The reliability_note must inform the model (and the user) that
    claims data is intentionally absent, so downstream answers can be
    properly hedged."""
    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, _selected_john())
    result = get_patient_clinical_data(ctx, ProceduresQuery())
    assert result.reliability_note is not None
    note = result.reliability_note.lower()
    assert "claims" in note
    assert "suppress" in note or "no patient identifier" in note


def test_procedures_filtered_by_cpt(synthetic_db):
    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, _selected_john())
    q = ProceduresQuery(cpt_codes=["45378"])
    result = get_patient_clinical_data(ctx, q)
    assert len(result.items) == 1


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


def test_surgeries_returns_data_present(synthetic_db):
    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, _selected_john())
    result = get_patient_clinical_data(ctx, SurgeriesQuery())
    assert result.domain == "surgeries"
    assert result.data_availability == "data_present"
    assert all(isinstance(i, SurgeryItem) for i in result.items)


def test_surgeries_excludes_non_surgical_cpt(synthetic_db):
    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, _selected_john())
    result = get_patient_clinical_data(ctx, SurgeriesQuery())
    codes = {i.code for i in result.items}
    assert "71046" not in codes
    assert "LOC-X-1" not in codes


def test_surgeries_performing_provider_ambiguous(synthetic_db):
    """Knee arthroplasty matches two encounters on same day → both providers listed, flagged ambiguous."""
    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, _selected_john())
    result = get_patient_clinical_data(ctx, SurgeriesQuery())
    knee = [i for i in result.items if i.code == "27447"]
    assert len(knee) == 1
    assert "Dr. Ortho" in knee[0].performing_provider
    assert "Dr. Anesthesia" in knee[0].performing_provider
    assert knee[0].provider_ambiguous is True


def test_surgeries_carries_reliability_note(synthetic_db):
    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, _selected_john())
    result = get_patient_clinical_data(ctx, SurgeriesQuery())
    assert result.reliability_note is not None
    assert "surgery" in result.reliability_note.lower() or "invasive" in result.reliability_note.lower()


def test_surgeries_date_range_filters(synthetic_db):
    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, _selected_john())
    q = SurgeriesQuery(date_range=DateRange(start=date(2025, 1, 1), end=date(2025, 12, 31)))
    result = get_patient_clinical_data(ctx, q)
    codes = {i.code for i in result.items}
    assert "27447" in codes


def test_allergies_returns_active_for_john_1962(synthetic_db):
    from agent.tools.get_patient_clinical_data import AllergiesQuery

    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, _selected_john())
    result = get_patient_clinical_data(ctx, AllergiesQuery())
    assert result.domain == "allergies"
    assert result.data_availability == "data_present"
    assert {i.allergy for i in result.items} == {"Penicillin", "Peanuts"}
    # severity/type/code carried through
    pen = next(i for i in result.items if i.allergy == "Penicillin")
    assert pen.category == "drug"
    assert pen.severity == "Severe"
    assert pen.code == "91936005"


def test_allergies_include_inactive_returns_resolved(synthetic_db):
    from agent.tools.get_patient_clinical_data import AllergiesQuery

    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, _selected_john())
    result = get_patient_clinical_data(ctx, AllergiesQuery(include_inactive=True))
    assert {i.allergy for i in result.items} == {"Penicillin", "Peanuts", "Latex"}


def test_allergies_nka_is_first_class_negative_assertion(synthetic_db):
    """Per epic #201: NO KNOWN ALLERGIES rows are surfaced as a
    negative_assertion flag on the result envelope, NOT as a regular item."""
    from agent.tools.get_patient_clinical_data import AllergiesQuery

    nka_patient = SelectedPatient(
        source_id="src-john-1971",
        display_name="John Smith",
        dob=date(1971, 8, 12),
        selected_at=datetime.now(),
        selection_origin="user_click",
    )
    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, nka_patient)
    result = get_patient_clinical_data(ctx, AllergiesQuery())
    assert result.data_availability == "data_present"
    assert result.negative_assertion is True
    assert result.items == []


def test_allergies_no_records_found_when_patient_has_no_allergy_rows(synthetic_db):
    from agent.tools.get_patient_clinical_data import AllergiesQuery

    no_allergy_patient = SelectedPatient(
        source_id="src-jane-1985",
        display_name="Jane Smith",
        dob=date(1985, 2, 20),
        selected_at=datetime.now(),
        selection_origin="user_click",
    )
    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, no_allergy_patient)
    result = get_patient_clinical_data(ctx, AllergiesQuery())
    assert result.data_availability == "no_records_found"
    assert result.negative_assertion is False
    assert result.items == []


def test_allergies_drug_med_conflict_emits_notes_to_agent(synthetic_db):
    """Per epic #201: when the patient has active medications and known
    drug allergens, the result emits a soft notes_to_agent flag describing
    potential overlap. Advisory only — not a CDS verdict."""
    from agent.tools.get_patient_clinical_data import AllergiesQuery

    sulfa_patient = SelectedPatient(
        source_id="src-mary-1956",
        display_name="Mary Jones",
        dob=date(1956, 3, 10),
        selected_at=datetime.now(),
        selection_origin="user_click",
    )
    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, sulfa_patient)
    result = get_patient_clinical_data(ctx, AllergiesQuery())
    assert result.notes_to_agent is not None
    # Soft signal must name the allergen AND the conflicting med.
    note = result.notes_to_agent.lower()
    assert "sulfa" in note
    assert "sulfamethoxazole" in note
    # Must explicitly disclaim CDS to keep the agent from over-promising.
    assert "advisory" in note or "not a clinical" in note or "not clinical" in note


def test_allergies_no_conflict_signal_when_meds_dont_overlap(synthetic_db):
    """John 1962 has Penicillin/Peanuts/Latex allergies but only Metformin +
    Azithromycin on the med list — neither is a beta-lactam, so no signal."""
    from agent.tools.get_patient_clinical_data import AllergiesQuery

    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, _selected_john())
    result = get_patient_clinical_data(ctx, AllergiesQuery())
    assert result.notes_to_agent is None


@pytest.mark.parametrize(
    "status",
    ["Discontinued", "No Longer Active", "SUSPENDED", "On Hold", "Completed"],
)
def test_allergy_advisory_excludes_known_inactive_medications(monkeypatch, status):
    """Verified-inactive meds do not alert (#298 goal)."""
    med = {
        "rxnorm_code": "10180",
        "med_name": "Sulfamethoxazole",
        "status": status,
        "status_date": "2024-06-01",
        "date_stopped": None,
    }
    monkeypatch.setattr(
        "agent.tools.get_patient_clinical_data.fetch_rows_across_source_ids",
        lambda *args, **kwargs: [med],
    )
    note = _maybe_drug_allergy_signal(
        MagicMock(),
        ["src-mary"],
        "",
        [{"allergy": "Sulfa", "type": "Drug allergy"}],
    )
    assert note is None


@pytest.mark.parametrize("status", ["Active", "Unknown", "administered", "undefined", "", None])
def test_allergy_advisory_includes_unknown_status_medications(monkeypatch, status):
    """#319 exclude-known-inactive: a med with unknown/blank/active status still
    participates in the advisory (soft advisory errs toward alerting), unlike the
    prior strict status=='active' rule that silently dropped ~5% of rows."""
    med = {
        "rxnorm_code": "10180",
        "med_name": "Sulfamethoxazole",
        "status": status,
        "date_stopped": None,
    }
    monkeypatch.setattr(
        "agent.tools.get_patient_clinical_data.fetch_rows_across_source_ids",
        lambda *args, **kwargs: [med],
    )
    note = _maybe_drug_allergy_signal(
        MagicMock(),
        ["src-mary"],
        "",
        [{"allergy": "Sulfa", "type": "Drug allergy"}],
    )
    assert note is not None


def test_allergy_advisory_excludes_explicitly_stopped_medication(monkeypatch):
    med = {
        "rxnorm_code": "10180",
        "med_name": "Sulfamethoxazole",
        "status": "Active",
        "date_stopped": "2024-06-01",
    }
    monkeypatch.setattr(
        "agent.tools.get_patient_clinical_data.fetch_rows_across_source_ids",
        lambda *args, **kwargs: [med],
    )
    assert (
        _maybe_drug_allergy_signal(
            MagicMock(),
            ["src-mary"],
            "",
            [{"allergy": "Sulfa", "type": "Drug allergy"}],
        )
        is None
    )


def test_admission_surfaces_admitting_event_diagnosing_clinician(synthetic_db):
    from agent.tools.get_patient_clinical_data import AdmissionsQuery

    selected = SelectedPatient(
        source_id="src-john-1971",
        display_name="John Smith",
        dob=date(1971, 8, 12),
        selected_at=datetime.now(),
        selection_origin="user_click",
    )
    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, selected)
    result = get_patient_clinical_data(ctx, AdmissionsQuery(facility_type="inpatient"))
    stays = [i for i in result.items if isinstance(i, AdmissionStay)]
    assert len(stays) == 1
    assert stays[0].diagnosing_clinician == "Admitting diagnosing clinician"


def test_allergies_reliability_note_names_source(synthetic_db):
    from agent.tools.get_patient_clinical_data import AllergiesQuery

    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, _selected_john())
    result = get_patient_clinical_data(ctx, AllergiesQuery())
    assert result.reliability_note is not None
    assert "federated_allergies_v" in result.reliability_note


def test_clinical_tool_sets_last_dataframe_on_deps(synthetic_db):
    """After get_patient_clinical_data returns, ctx.deps.last_dataframe
    should hold a pandas DataFrame of the result items."""
    import pandas as pd

    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, _selected_john())

    result = get_patient_clinical_data(ctx, DemographicsQuery())

    assert result.domain == "demographics"
    assert isinstance(ctx.deps.last_dataframe, pd.DataFrame)
    assert len(ctx.deps.last_dataframe) == len(result.items)


def test_clinical_tool_empty_result_sets_empty_dataframe(synthetic_db):
    """A no_records_found result should still set an (empty) DataFrame
    on deps so downstream tools see a definite signal rather than None."""
    import pandas as pd
    from agent.tools.get_patient_clinical_data import LabsQuery

    selected_no_data = SelectedPatient(
        source_id="src-nonexistent",
        display_name="Nobody",
        dob=None,
        selected_at=datetime.now(),
        selection_origin="user_click",
    )
    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, selected_no_data)

    result = get_patient_clinical_data(ctx, LabsQuery(date_range=None))

    assert result.data_availability == "no_records_found"
    assert isinstance(ctx.deps.last_dataframe, pd.DataFrame)
    assert len(ctx.deps.last_dataframe) == 0
