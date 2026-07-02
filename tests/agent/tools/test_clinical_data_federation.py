"""EMPI federation across sibling source_ids for patient-scoped tools.

Real patients hold their chart under several source_ids (one per contributing
source system, linked via internal_source_reference_v.patient_id). The tools
must resolve the selected source_id to the full federation set instead of
filtering on the single raw id, which returns [] and sends the agent flailing
— e.g. a prod patient with all 202 encounters under an empi_rank=2 sibling.

Two layers here:
- synthetic_db integration tests: the fixture seeds src-john-1962 (rank 1) and
  src-john-1962-alt (rank 2, no clinical rows of its own), so selecting the
  alt sibling must surface the rows stored under the canonical id.
- DB-free stub tests: route fetch_all on SQL text / bind params (per Chiron's
  test_get_clinical_data_federation.py precedent) to pin dedupe, merge-sort,
  global most_recent_only, and the admissions non-double-federation contract.
"""

from datetime import date, datetime
from unittest.mock import MagicMock

from agent.deps import AgentDeps, SelectedPatient
from agent.db.analytics_adapter import AnalyticsDbAdapter
from agent.tools.get_patient_clinical_data import (
    get_patient_clinical_data,
    AdmissionsQuery,
    AllergiesQuery,
    EncountersQuery,
    LabsQuery,
)
from agent.tools.list_patient_documents import (
    list_patient_documents,
    DocumentIndexQuery,
)


def _deps(adapter, selected: SelectedPatient | None) -> AgentDeps:
    return AgentDeps(
        user_id=1,
        user_role=MagicMock(value=1),
        session_id="s1",
        selected_patient=selected,
        last_dataframe=None,
        last_sql=None,
        last_query_meta=None,
        analytics_db=adapter,
        rag=None,
        sqlite_session=None,
        run_logger=MagicMock(),
    )


def _selected(source_id: str) -> SelectedPatient:
    return SelectedPatient(
        source_id=source_id,
        display_name="John Smith",
        dob=date(1962, 5, 1),
        selected_at=datetime.now(),
        selection_origin="user_click",
    )


def _ctx(adapter, source_id: str = "SRC-A") -> MagicMock:
    ctx = MagicMock()
    ctx.deps = _deps(adapter, _selected(source_id))
    return ctx


# --- Integration: synthetic fixture already seeds a rank-2 sibling ----


def test_sibling_source_id_surfaces_canonical_rows(synthetic_db):
    """Selecting src-john-1962-alt (empi_rank=2, no rows of its own) must
    return the encounters stored under canonical sibling src-john-1962."""
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    ctx = _ctx(adapter, "src-john-1962-alt")
    result = get_patient_clinical_data(ctx, EncountersQuery())
    assert result.data_availability == "data_present"
    assert len(result.items) == 4


def test_unknown_source_id_falls_back_to_single_id_query(synthetic_db):
    """A sid absent from internal_source_reference_v (e.g. minimal fixtures)
    must fall back to the plain single-id query, not error."""
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    ctx = _ctx(adapter, "src-nobody")
    result = get_patient_clinical_data(ctx, EncountersQuery())
    assert result.data_availability == "no_records_found"
    assert result.items == []


# --- DB-free stubs: SRC-A (rank 1) + SRC-B (rank 2, duplicated across two
# source_names, as prod EMPI data does) are the same patient --------------


_SID_ROWS = [
    {"source_id": "SRC-A", "empi_rank": 1, "source_name": "ehr"},
    {"source_id": "SRC-B", "empi_rank": 2, "source_name": "hie"},
    {"source_id": "SRC-B", "empi_rank": 2, "source_name": "claims"},
]


class _RoutingAdapter:
    """fetch_all stub. Routes federated_*_v queries to per-sid row fixtures;
    answers identity-resolution SQL with _SID_ROWS (or failure when sid_rows
    is None). Records (view, params) per domain query for dedupe assertions."""

    dialect = "sqlite"
    schema_prefix = ""

    def __init__(self, rows_by_view: dict[str, dict[str, list[dict]]], sid_rows=_SID_ROWS):
        self.rows_by_view = rows_by_view
        self.sid_rows = sid_rows
        self.queries: list[tuple[str, dict]] = []

    def fetch_all(self, sql: str, params: dict | None = None) -> list[dict]:
        params = params or {}
        # Domain views first: the adt SQL joins internal_source_reference_v
        # itself, so the ISR check below must not shadow it.
        for view, by_sid in self.rows_by_view.items():
            if view in sql:
                self.queries.append((view, dict(params)))
                return list(by_sid.get(params.get("source_id", ""), []))
        if "internal_source_reference_v" in sql:
            if "internal_patient_id" in params:
                return list(self.sid_rows or [])
            return [] if self.sid_rows is None else [{"internal_patient_id": 7}]
        raise AssertionError(f"unexpected SQL in stub: {sql[:120]}")


def _encounter_row(sid: str, event_datetime: str | None) -> dict:
    return {
        "source_id": sid,
        "encounter_id": f"ENC-{sid}-{event_datetime or 'null'}",
        "type": "office_visit",
        "status": "completed",
        "event_datetime": event_datetime,
        "location": None,
        "rendering_provider": None,
        "facility_name": "Clinic",
        "place_of_service": "11",
    }


def test_rows_under_sibling_sid_are_returned():
    adapter = _RoutingAdapter({"federated_encounters_v": {"SRC-B": [_encounter_row("SRC-B", "2025-08-07 09:00")]}})
    result = get_patient_clinical_data(_ctx(adapter), EncountersQuery())
    assert result.data_availability == "data_present"
    assert [i.encounter_id for i in result.items] == ["ENC-SRC-B-2025-08-07 09:00"]


def test_each_unique_source_id_queried_once():
    adapter = _RoutingAdapter({"federated_encounters_v": {}})
    get_patient_clinical_data(_ctx(adapter), EncountersQuery())
    sids = [p["source_id"] for _, p in adapter.queries]
    assert sids == ["SRC-A", "SRC-B"]  # deduped: SRC-B once despite 2 EMPI rows


def test_merged_rows_sorted_date_desc_nulls_last():
    adapter = _RoutingAdapter(
        {
            "federated_encounters_v": {
                "SRC-A": [
                    _encounter_row("SRC-A", "2024-05-01 10:00"),
                    _encounter_row("SRC-A", None),
                ],
                "SRC-B": [_encounter_row("SRC-B", "2025-08-07 09:00")],
            }
        }
    )
    ctx = _ctx(adapter)
    result = get_patient_clinical_data(ctx, EncountersQuery())
    assert [i.event_datetime for i in result.items] == [
        "2025-08-07 09:00",
        "2024-05-01 10:00",
        None,
    ]
    # last_dataframe reflects the merged item set.
    assert len(ctx.deps.last_dataframe) == 3


def _lab_row(sid: str, event_datetime: str, value: str) -> dict:
    return {
        "source_id": sid,
        "code": "4548-4",
        "code_type": "LOINC",
        "name": "Hemoglobin A1c",
        "result": value,
        "clean_result": value,
        "unit": "%",
        "event_datetime": event_datetime,
        "service_provider": None,
    }


def test_most_recent_only_is_global_across_source_ids():
    """Per-sid LIMIT 1 leaves one candidate per source; only the globally
    newest row may survive the merge."""
    adapter = _RoutingAdapter(
        {
            "federated_results_v": {
                "SRC-A": [_lab_row("SRC-A", "2024-03-01 08:00", "6.1")],
                "SRC-B": [_lab_row("SRC-B", "2025-06-01 08:00", "7.2")],
            }
        }
    )
    result = get_patient_clinical_data(_ctx(adapter), LabsQuery(most_recent_only=True))
    assert len(result.items) == 1
    assert result.items[0].clean_result == "7.2"


def test_admissions_query_is_not_double_federated():
    """adt.py already resolves identity through the EMPI join — the tool must
    pass the single selected sid and query exactly once."""
    adapter = _RoutingAdapter({"federated_adt_v": {}})
    result = get_patient_clinical_data(_ctx(adapter), AdmissionsQuery())
    assert result.data_availability == "no_records_found"
    adt_queries = [p for v, p in adapter.queries if v == "federated_adt_v"]
    assert len(adt_queries) == 1
    assert adt_queries[0]["source_id"] == "SRC-A"


def test_drug_allergy_signal_sees_meds_under_sibling_sid():
    """Penicillin allergy recorded under SRC-A; the conflicting amoxicillin
    prescription lives under SRC-B. The advisory must still fire."""
    allergy_row = {
        "source_id": "SRC-A",
        "code": "91936005",
        "code_type": "SNOMED",
        "allergy": "Penicillin",
        "type": "Drug allergy",
        "severity": "Severe",
        "status": "Active",
        "onset_date": None,
        "event_datetime": "2024-01-01 00:00",
        "reaction": "Hives",
        "comments": None,
    }
    med_row = {
        "source_id": "SRC-B",
        "ndc_code": None,
        "rxnorm_code": "723",
        "med_name": "Amoxicillin",
        "date_prescribed": "2026-05-01 00:00",
        "status": "active",
    }
    adapter = _RoutingAdapter(
        {
            "federated_allergies_v": {"SRC-A": [allergy_row]},
            "federated_meds_v": {"SRC-B": [med_row]},
        }
    )
    result = get_patient_clinical_data(_ctx(adapter), AllergiesQuery())
    assert result.notes_to_agent is not None
    assert "Amoxicillin" in result.notes_to_agent


def test_documents_federated_and_sorted():
    def _doc(sid: str, event_datetime: str | None, name: str) -> dict:
        return {
            "source_id": sid,
            "event_datetime": event_datetime,
            "name": name,
            "mnemonic": None,
            "status": "final",
            "encounter_id": None,
            "place_of_service": None,
            "location_name": None,
        }

    adapter = _RoutingAdapter(
        {
            "federated_documents_v": {
                "SRC-A": [_doc("SRC-A", "2024-02-02 08:00", "Office Note")],
                "SRC-B": [_doc("SRC-B", "2025-08-07 09:00", "Discharge Summary")],
            }
        }
    )
    result = list_patient_documents(_ctx(adapter), DocumentIndexQuery())
    assert [d.name for d in result.documents] == ["Discharge Summary", "Office Note"]


def test_find_patient_related_source_ids_deduped():
    """internal_source_reference_v holds one row per (source_id, source_name),
    so the same sid recurs in related rows — dedupe is load-bearing."""
    from agent.tools.find_patient import find_patient, PatientSearchQuery

    class _FindStub:
        dialect = "sqlite"
        schema_prefix = ""

        def fetch_all(self, sql: str, params: dict | None = None) -> list[dict]:
            params = params or {}
            if "internal_patient_id" in params:  # related_source_ids_sql
                return [
                    {"source_id": "SRC-B", "empi_rank": 2, "source_name": "hie"},
                    {"source_id": "SRC-B", "empi_rank": 2, "source_name": "claims"},
                ]
            return [
                {
                    "internal_patient_id": 7,
                    "source_id": "SRC-A",
                    "display_name": "Pat Example",
                    "first_name": "Pat",
                    "last_name": "Example",
                    "dob": None,
                    "age": 56,
                    "most_recent_activity": None,
                    "practice_name": None,
                }
            ]

    ctx = MagicMock()
    ctx.deps = _deps(_FindStub(), None)
    results = find_patient(ctx, PatientSearchQuery(last_name="Example"))
    assert results.matches[0].related_source_ids == ["SRC-B"]
