"""search_patients_by_criteria — cohort / population tool.

Phase 4 design §3.1. Operates WITHOUT a selected patient (unlike every
other clinical-data tool); the docstring + system prompt route population
questions here instead of through find_patient + get_patient_clinical_data.

The tool body is added in the next task; this module defines the shapes
the dataframe adapter (Task 3) and the test fixtures (Task 4) reference.
"""

from __future__ import annotations
from datetime import date
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent.result_compaction import CompactingListResult


class DateRange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: Optional[date] = None
    end: Optional[date] = None


class CohortCriteria(BaseModel):
    # extra='forbid': weak models sometimes pass a free-text top-level
    # field like `query` or `description`. Reject it so pydantic-ai
    # feeds the validation error back as a retry.
    model_config = ConfigDict(extra="forbid")

    diagnosis_codes: Optional[List[str]] = None
    diagnosis_date_range: Optional[DateRange] = None
    medication_rxnorm_codes: Optional[List[str]] = None
    condition_text: Optional[str] = None
    age_min: Optional[int] = None
    age_max: Optional[int] = None
    gender: Optional[Literal["M", "F", "U"]] = None
    facility: Optional[str] = None
    last_visit_after: Optional[date] = None
    last_visit_before: Optional[date] = None
    sample_size: int = Field(default=20, ge=0, le=100)

    @model_validator(mode="after")
    def _require_at_least_one_criterion(self) -> "CohortCriteria":
        # diagnosis_date_range alone is a no-op (only used to narrow
        # diagnosis_codes), so it does not count as a criterion.
        if not any(
            (
                self.diagnosis_codes,
                self.medication_rxnorm_codes,
                self.condition_text,
                self.age_min is not None,
                self.age_max is not None,
                self.gender,
                self.facility,
                self.last_visit_after,
                self.last_visit_before,
            )
        ):
            raise ValueError(
                "search_patients_by_criteria requires at least one criterion "
                "(diagnosis_codes, medication_rxnorm_codes, condition_text, "
                "age_min/age_max, gender, facility, last_visit_after, or "
                "last_visit_before). Do not call without a filter."
            )
        return self


class PatientMatch(BaseModel):
    source_id: str
    display_name: str
    age: Optional[int] = None
    gender: Optional[str] = None
    last_date_of_visit: Optional[date] = None
    practice_name: Optional[str] = None


class CohortResult(CompactingListResult):
    _list_field = "sample"

    total_count: int
    sample: List[PatientMatch]
    data_availability: Literal["data_present", "no_records_found", "error"]
    truncated: bool = False
    reliability_note: Optional[str] = None
    notes_to_agent: Optional[str] = None


from pydantic_ai import RunContext
from agent.dataframe_adapters import cohort_result_to_df
from agent.db.queries.cohort import cohort_sql
from agent.deps import AgentDeps


_RELIABILITY_DX = (
    "ICD-10 coverage in problems ~57%; SNOMED ~25%. Counts are best-effort. "
    "Some matches may come from claims data refreshed monthly; clinical events refresh bi-weekly."
)
_RELIABILITY_RX = (
    "RxNorm + NDC coverage in medications is ~100%, but cohort counts are still "
    "subject to data-refresh cadence. Some matches may come from claims data refreshed monthly."
)


def search_patients_by_criteria(ctx: RunContext[AgentDeps], criteria: CohortCriteria) -> CohortResult:
    """Find patients matching demographic and clinical criteria.

    Use this tool for questions over a POPULATION of patients — "how many",
    "count of", "find/list patients with", "show me N of...". Do NOT call
    find_patient first; this tool operates without a selected patient.

    If the question is about a SPECIFIC patient (the user named them, said
    "he/she", "the patient", or "their X"), use find_patient +
    get_patient_clinical_data instead. Distinguishing signal: pluralization
    without a name = population; "his/her/the patient's" = specific-patient.
    "How many patients have diabetes?" is population. "How many medications
    is John taking?" is specific-patient.

    Criteria fields:
      - diagnosis_codes: ICD-10 or SNOMED codes; matched in metric_federated_data_v.
      - diagnosis_date_range: optional DateRange narrowing diagnosis events.
      - medication_rxnorm_codes: RxNorm or NDC codes; matched in metric_federated_data_v.
      - condition_text: free-text fallback when codes are not known; LIKE matches
        the denormalized conditions string. Prefer diagnosis_codes when available.
      - age_min / age_max: integer bounds.
      - gender: "M", "F", or "U".
      - facility: substring match against practice_name (case-insensitive).
      - last_visit_after / last_visit_before: date bounds on last_date_of_visit.
      - sample_size: 0-100; default 20. Set to 0 for count-only.

    Returns CohortResult with total_count + sample. The sample is truncated
    to sample_size; total_count carries the full population. When a code-based
    filter is set, reliability_note carries a coverage caveat that the agent
    MUST surface to the user.
    """
    adapter = ctx.deps.analytics_db
    schema_prefix = getattr(adapter, "schema_prefix", "")

    sql, params = cohort_sql(criteria, schema_prefix=schema_prefix)
    rows = adapter.fetch_all(sql, params)

    if not rows:
        result = CohortResult(
            total_count=0,
            sample=[],
            data_availability="no_records_found",
        )
        ctx.deps.last_dataframe = cohort_result_to_df(result)
        return result

    total_count = int(rows[0]["total_count"])
    # cohort_sql applies LIMIT sample_size+1 as a truncation sentinel
    # (see agent/db/queries/cohort.py); mirrors find_patient.truncated.
    truncated = len(rows) > criteria.sample_size

    sample_rows = rows[: criteria.sample_size] if criteria.sample_size > 0 else []
    sample = [
        PatientMatch(
            source_id=r["source_id"],
            display_name=r["display_name"],
            age=r.get("age"),
            gender=r.get("gender"),
            last_date_of_visit=(
                r["last_date_of_visit"]
                if isinstance(r.get("last_date_of_visit"), date)
                else (date.fromisoformat(r["last_date_of_visit"]) if r.get("last_date_of_visit") else None)
            ),
            practice_name=r.get("practice_name"),
        )
        for r in sample_rows
    ]

    reliability = None
    if criteria.diagnosis_codes:
        reliability = _RELIABILITY_DX
    elif criteria.medication_rxnorm_codes:
        reliability = _RELIABILITY_RX

    result = CohortResult(
        total_count=total_count,
        sample=sample,
        data_availability="data_present",
        truncated=truncated,
        reliability_note=reliability,
    )
    ctx.deps.last_dataframe = cohort_result_to_df(result)
    return result
