"""Each clinical-data tool must read schema_prefix off the adapter and
forward it to the SQL template. Otherwise the prod Redshift queries hit
unqualified `federated_*_v` instead of `dw.federated_*_v`.
"""

from __future__ import annotations

from datetime import date, datetime
from unittest.mock import MagicMock

import pytest
from pydantic_ai import RunContext

from agent.deps import AgentDeps, SelectedPatient
from agent.tools.find_patient import find_patient, PatientSearchQuery
from agent.tools.list_patient_documents import list_patient_documents, DocumentIndexQuery
from agent.tools.get_patient_clinical_data import (
    get_patient_clinical_data,
    DemographicsQuery,
    EncountersQuery,
    LabsQuery,
    DiagnosesQuery,
    MedicationsQuery,
    ImmunizationsQuery,
    ProceduresQuery,
    ImagingQuery,
)


class _CapturingAdapter:
    """Minimal AnalyticsDbAdapter stand-in. Records fetch_all's first
    argument so tests can inspect the SQL the tool generated."""

    def __init__(self, schema: str = ""):
        self.schema = schema
        self.captured_sqls: list[str] = []

    @property
    def schema_prefix(self) -> str:
        s = (self.schema or "").rstrip(".")
        return f"{s}." if s else ""

    def fetch_all(self, sql: str, params=None) -> list[dict]:
        self.captured_sqls.append(sql)
        return []  # all tools tolerate empty result


def _ctx(adapter, *, with_selection: bool):
    deps = MagicMock(spec=AgentDeps)
    deps.analytics_db = adapter
    if with_selection:
        deps.selected_patient = SelectedPatient(
            source_id="src-x",
            display_name="X",
            dob=date(1990, 1, 1),
            selected_at=datetime(2026, 5, 6),
            selection_origin="user_click",
        )
    else:
        deps.selected_patient = None
    return MagicMock(spec=RunContext, deps=deps)


def test_find_patient_forwards_schema_prefix():
    adapter = _CapturingAdapter(schema="dw")
    ctx = _ctx(adapter, with_selection=False)
    find_patient(ctx, PatientSearchQuery(last_name="Smith"))
    assert adapter.captured_sqls, "fetch_all was never called"
    # find_patient queries internal_patient_profile_v and internal_source_reference_v
    assert any("dw.internal_patient_profile_v" in s for s in adapter.captured_sqls)
    assert any("dw.internal_source_reference_v" in s for s in adapter.captured_sqls)


def test_list_patient_documents_forwards_schema_prefix():
    adapter = _CapturingAdapter(schema="dw")
    ctx = _ctx(adapter, with_selection=True)
    list_patient_documents(ctx, DocumentIndexQuery())
    assert adapter.captured_sqls
    assert "dw.federated_documents_v" in adapter.captured_sqls[0]


@pytest.mark.parametrize(
    "query,expected_view",
    [
        (DemographicsQuery(), "dw.federated_demographic_v"),
        (EncountersQuery(), "dw.federated_encounters_v"),
        (LabsQuery(), "dw.federated_results_v"),
        (DiagnosesQuery(), "dw.federated_problems_v"),
        (MedicationsQuery(), "dw.federated_meds_v"),
        (ImmunizationsQuery(), "dw.federated_vaccination_v"),
        (ProceduresQuery(), "dw.federated_orders_v"),  # one of three; presence sufficient
        (ImagingQuery(), "dw.federated_orders_v"),  # one of two; presence sufficient
    ],
)
def test_get_patient_clinical_data_forwards_schema_prefix(query, expected_view):
    adapter = _CapturingAdapter(schema="dw")
    ctx = _ctx(adapter, with_selection=True)
    get_patient_clinical_data(ctx, query)
    assert adapter.captured_sqls, f"no SQL captured for {type(query).__name__}"
    combined = "\n".join(adapter.captured_sqls)
    assert expected_view in combined


def test_default_empty_schema_does_not_break_anything():
    """Sanity: when schema is empty, SQL should be unchanged from
    pre-schema behavior — no leading dots, no malformed prefixes."""
    adapter = _CapturingAdapter(schema="")
    ctx = _ctx(adapter, with_selection=True)
    get_patient_clinical_data(ctx, DemographicsQuery())
    sql = adapter.captured_sqls[0]
    assert "FROM federated_demographic_v" in sql
    assert "FROM .federated_demographic_v" not in sql
