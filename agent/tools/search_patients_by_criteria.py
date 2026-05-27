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

from agent.db.queries.cohort_breakdown import BreakdownDimension
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
    zip_code: Optional[str] = None  # 5-digit ZIP, exact match on p.zip_code
    city: Optional[str] = None  # case-insensitive substring on p.city
    state: Optional[str] = None  # 2-letter USPS code, exact match on uppercased p.state
    sample_size: int = Field(default=20, ge=0, le=100)
    breakdown: List[BreakdownDimension] = Field(default_factory=list)

    @model_validator(mode="after")
    def _require_at_least_one_criterion(self) -> "CohortCriteria":
        # A diagnosis_date_range with a start or end bound is a standalone
        # criterion ("any diagnosis in this window"). An all-None DateRange
        # carries no filter and does not count.
        dr_active = self.diagnosis_date_range is not None and (
            self.diagnosis_date_range.start or self.diagnosis_date_range.end
        )
        if not any(
            (
                self.diagnosis_codes,
                dr_active,
                self.medication_rxnorm_codes,
                self.condition_text,
                self.age_min is not None,
                self.age_max is not None,
                self.gender,
                self.facility,
                self.last_visit_after,
                self.last_visit_before,
                self.zip_code,
                self.city,
                self.state,
            )
        ):
            raise ValueError(
                "search_patients_by_criteria requires at least one criterion "
                "(diagnosis_codes, medication_rxnorm_codes, condition_text, "
                "age_min/age_max, gender, facility, last_visit_after, "
                "last_visit_before, zip_code, city, or state). "
                "Do not call without a filter."
            )
        return self


class PatientMatch(BaseModel):
    source_id: str
    display_name: str
    age: Optional[int] = None
    gender: Optional[str] = None
    last_date_of_visit: Optional[date] = None
    practice_name: Optional[str] = None


class BreakdownBucket(BaseModel):
    bucket_label: str
    patient_count: int


class CohortResult(CompactingListResult):
    _list_field = "sample"

    total_count: int
    sample: List[PatientMatch]
    data_availability: Literal["data_present", "no_records_found", "error"]
    truncated: bool = False
    reliability_note: Optional[str] = None
    notes_to_agent: Optional[str] = None
    buckets: List[BreakdownBucket] = Field(default_factory=list)
    non_additive: bool = False
    generated_sql: Optional[str] = None
    breakdown_status: Optional[
        Literal["single_dimension", "unsupported_multi_dimension", "missing_diagnosis_anchor"]
    ] = None


from pydantic_ai import RunContext
from pydantic_ai.exceptions import ModelRetry
from sqlalchemy.exc import SQLAlchemyError

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
_RELIABILITY_GEO = (
    "Geographic filters use structured columns on internal_patient_profile_v "
    "(zip_code, city, state). zip_code coverage is ~100%. City strings vary "
    "in casing and abbreviation; substring matching handles common cases but "
    "may miss heavily abbreviated forms. State filtering matches 'NY'-style "
    "codes; addresses recorded as 'NEW YORK' may be missed."
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
      - diagnosis_date_range: DateRange (start/end, inclusive) over diagnosis
        start_date. Works WITH diagnosis_codes (narrows those codes to the
        window) OR ALONE — a bare range counts patients with ANY ICD-10/SNOMED
        diagnosis in the window. Use it alone for "how many patients had a
        diagnosis in <period>". Do NOT fake this with age_min/age_max: an
        age band does not filter by diagnosis date and yields the whole
        population.
      - medication_rxnorm_codes: RxNorm or NDC codes; matched in metric_federated_data_v.
      - condition_text: free-text fallback when codes are not known; LIKE matches
        the denormalized conditions string. Prefer diagnosis_codes when available.
      - age_min / age_max: integer bounds.
      - gender: "M", "F", or "U".
      - facility: substring match against practice_name (case-insensitive).
      - last_visit_after / last_visit_before: date bounds on last_date_of_visit.
      - zip_code: 5-digit ZIP code, exact match against internal_patient_profile_v.zip_code (~100% coverage).
      - city: city name, case-insensitive substring match (city strings vary by source).
      - state: 2-letter USPS code, exact match on uppercased input. May miss long-form 'NEW YORK'.
      - sample_size: 0-100; default 20. Set to 0 for count-only.

    Returns CohortResult with total_count + sample. The sample is truncated
    to sample_size; total_count carries the full population. When a code-based
    filter is set, reliability_note carries a coverage caveat that the agent
    MUST surface to the user.
    """
    adapter = ctx.deps.analytics_db
    schema_prefix = getattr(adapter, "schema_prefix", "")

    sql, params = cohort_sql(criteria, schema_prefix=schema_prefix)
    try:
        rows = adapter.fetch_all(sql, params)
    except SQLAlchemyError as exc:
        # Statement timeout / undefined column / connection drop / etc.
        # Without this catch the exception bubbled up unwrapped through
        # pydantic-ai and crashed the entire agent stream (live regression
        # 2026-05-13 on 'how many people in NY have diabetes' — broad
        # cohort scan hit the 30s curated-query timeout). ModelRetry lets
        # the LLM see the failure and either narrow the criteria
        # (smaller geographic filter, age bound, etc.) or report the
        # limitation to the user.
        raise ModelRetry(
            f"Cohort query failed: {exc}. Try narrowing the criteria "
            "(add a smaller geographic filter, an age range, or a more "
            "specific date window) or report this limitation to the user."
        ) from exc

    # Compute reliability_note up front so the no_records path also surfaces it
    # — geo-filter misses are most informative when no rows came back.
    reliability_parts: list[str] = []
    _dr = criteria.diagnosis_date_range
    _dr_active = _dr is not None and (_dr.start or _dr.end)
    if criteria.diagnosis_codes or _dr_active:
        reliability_parts.append(_RELIABILITY_DX)
    elif criteria.medication_rxnorm_codes:
        reliability_parts.append(_RELIABILITY_RX)
    if criteria.zip_code or criteria.city or criteria.state:
        reliability_parts.append(_RELIABILITY_GEO)
    reliability = " ".join(reliability_parts) if reliability_parts else None

    # Count-only fast path: cohort_sql emits a single COUNT(DISTINCT) row
    # when sample_size==0. Surface total_count even if it's zero — the
    # absence of a row would have been caught above as a SQL error.
    if int(criteria.sample_size) == 0:
        total_count = int(rows[0]["total_count"]) if rows else 0
        availability = "data_present" if total_count > 0 else "no_records_found"
        result = CohortResult(
            total_count=total_count,
            sample=[],
            data_availability=availability,
            reliability_note=reliability,
        )
        ctx.deps.last_dataframe = cohort_result_to_df(result)
        return result

    if not rows:
        result = CohortResult(
            total_count=0,
            sample=[],
            data_availability="no_records_found",
            reliability_note=reliability,
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

    result = CohortResult(
        total_count=total_count,
        sample=sample,
        data_availability="data_present",
        truncated=truncated,
        reliability_note=reliability,
    )
    ctx.deps.last_dataframe = cohort_result_to_df(result)
    return result
