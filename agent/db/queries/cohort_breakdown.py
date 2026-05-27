"""Single-dimension breakdown SQL for search_patients_by_criteria.

Design: docs/superpowers/specs/2026-05-27-cohort-breakdown-dimensions-design.md

Builds a GROUP BY query over the same filtered cohort that cohort_sql
produces, counting COUNT(DISTINCT isr.source_id) per bucket. Time axes
(diagnosis month/quarter/year) are multi-valued: a patient appears in
every bucket they were active, so buckets overlap and do NOT sum to the
population total (non_additive=True). Gender / age band are single-valued
and additive.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

from agent.db.queries.cohort import (
    _build_non_diagnosis_filters,
    _diagnosis_event_where,
)


class BreakdownDimension(str, Enum):
    DIAGNOSIS_MONTH = "diagnosis_month"
    DIAGNOSIS_QUARTER = "diagnosis_quarter"
    DIAGNOSIS_YEAR = "diagnosis_year"
    GENDER = "gender"
    AGE_BAND = "age_band"


@dataclass(frozen=True)
class BucketSpec:
    group_expr: str  # SQL expression used in SELECT + GROUP BY, aliased to bucket_label
    non_additive: bool
    requires_diagnosis_anchor: bool


_AGE_BAND_CASE = (
    "CASE "
    "WHEN p.age IS NULL THEN 'Unknown' "
    "WHEN p.age < 18 THEN '0-17' "
    "WHEN p.age < 35 THEN '18-34' "
    "WHEN p.age < 50 THEN '35-49' "
    "WHEN p.age < 65 THEN '50-64' "
    "ELSE '65+' END"
)

# null gender and the 'U' (unknown) sentinel both collapse to one bucket.
_GENDER_CASE = "CASE WHEN p.gender IN ('M', 'F') THEN p.gender ELSE 'Unknown' END"


def _time_expr(unit: str, dialect: str) -> str:
    """Stable string label for a diagnosis-date bucket, per dialect.

    unit: 'month' | 'quarter' | 'year'. Operates on dx.start_date (the
    diagnosis-event subquery alias).
    """
    col = "dx.start_date"
    if dialect == "sqlite":
        if unit == "month":
            return f"strftime('%Y-%m', {col})"
        if unit == "year":
            return f"strftime('%Y', {col})"
        # quarter: 'YYYY-Qn'
        return f"strftime('%Y', {col}) || '-Q' || ((CAST(strftime('%m', {col}) AS INTEGER) + 2) / 3)"
    if dialect in ("postgres", "redshift"):
        if unit == "month":
            return f"TO_CHAR(DATE_TRUNC('month', {col}), 'YYYY-MM')"
        if unit == "year":
            return f"TO_CHAR({col}, 'YYYY')"
        return f"TO_CHAR(DATE_TRUNC('quarter', {col}), 'YYYY-\"Q\"Q')"
    raise ValueError(f"Unsupported dialect for breakdown: {dialect!r}")


def breakdown_bucket(dimension: BreakdownDimension, dialect: str) -> BucketSpec:
    """Resolve a dimension to its SQL bucket expression for the given dialect."""
    if dimension == BreakdownDimension.GENDER:
        return BucketSpec(_GENDER_CASE, non_additive=False, requires_diagnosis_anchor=False)
    if dimension == BreakdownDimension.AGE_BAND:
        return BucketSpec(_AGE_BAND_CASE, non_additive=False, requires_diagnosis_anchor=False)
    unit = {
        BreakdownDimension.DIAGNOSIS_MONTH: "month",
        BreakdownDimension.DIAGNOSIS_QUARTER: "quarter",
        BreakdownDimension.DIAGNOSIS_YEAR: "year",
    }[dimension]
    return BucketSpec(_time_expr(unit, dialect), non_additive=True, requires_diagnosis_anchor=True)


class MissingDiagnosisAnchorError(ValueError):
    """A diagnosis-time breakdown was requested without a diagnosis anchor."""


def cohort_breakdown_sql(
    criteria,
    dimension: BreakdownDimension,
    schema_prefix: str = "",
    dialect: str = "postgres",
) -> tuple[str, str, dict]:
    """Build (bucket_sql, total_sql, params) for a single-dimension breakdown.

    bucket_sql: SELECT <bucket_expr> AS bucket_label,
                COUNT(DISTINCT isr.source_id) AS patient_count ... GROUP BY 1 ORDER BY 1
    total_sql:  SELECT COUNT(DISTINCT isr.source_id) AS total_count over the same cohort
                (same grain so additive buckets sum to total).

    Raises MissingDiagnosisAnchorError when a time dimension is requested
    but no diagnosis criterion is active.
    """
    spec = breakdown_bucket(dimension, dialect)

    dx = _diagnosis_event_where(criteria)
    if spec.requires_diagnosis_anchor and dx is None:
        raise MissingDiagnosisAnchorError(
            "A diagnosis-time breakdown needs a diagnosis anchor: pass diagnosis_codes or a diagnosis_date_range."
        )

    params: dict = {}
    join_clauses: list[str] = []
    where_clauses: list[str] = ["1=1"]

    if dx is not None:
        dx_filter, dx_params = dx
        params.update(dx_params)
        if spec.requires_diagnosis_anchor:
            # Event-level join: keep start_date so we can bucket on it.
            join_clauses.append(
                f"JOIN (SELECT patient_id, start_date FROM {schema_prefix}metric_federated_data_v "
                f"WHERE {' AND '.join(dx_filter)}) dx ON dx.patient_id = p.patient_id"
            )
        else:
            # Diagnosis used only as a filter; DISTINCT keeps the cohort one row/patient.
            join_clauses.append(
                f"JOIN (SELECT DISTINCT patient_id FROM {schema_prefix}metric_federated_data_v "
                f"WHERE {' AND '.join(dx_filter)}) dx ON dx.patient_id = p.patient_id"
            )

    other_joins, other_where, other_params = _build_non_diagnosis_filters(criteria, schema_prefix)
    join_clauses.extend(other_joins)
    where_clauses.extend(other_where)
    params.update(other_params)

    join_block = "\n        ".join(join_clauses)
    where_block = " AND ".join(where_clauses)
    from_block = (
        f"FROM {schema_prefix}internal_patient_profile_v p\n"
        f"        JOIN {schema_prefix}internal_source_reference_v isr\n"
        f"          ON isr.patient_id = p.patient_id AND isr.empi_rank != 99\n"
        f"        {join_block}\n"
        f"        WHERE {where_block}"
    )

    bucket_sql = (
        f"SELECT {spec.group_expr} AS bucket_label, "
        f"COUNT(DISTINCT isr.source_id) AS patient_count\n"
        f"        {from_block}\n"
        f"        GROUP BY {spec.group_expr}\n"
        f"        ORDER BY {spec.group_expr}"
    )
    total_sql = f"SELECT COUNT(DISTINCT isr.source_id) AS total_count\n        {from_block}"
    return bucket_sql, total_sql, params
