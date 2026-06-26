"""SQL template for the cohort tool.

Phase 4 design §3.2 — single SELECT over internal_patient_profile_v +
internal_source_reference_v, with optional inner-join subqueries on
metric_federated_data_v for diagnosis_codes and medication_rxnorm_codes
filters. COUNT(*) OVER () returns the population total in one round trip;
LIMIT sample_size+1 lets the caller detect that the cohort exceeds the
sample.

Positional placeholders (:dx_0, :dx_1, ...) are used for IN-list filters
because the project's existing SQLAlchemy text() pattern does NOT use
expanding bindparam — see agent/db/queries/labs.py for the same idiom.
"""

from __future__ import annotations
from typing import Tuple


# WNY-relevant US state aliases. The warehouse's state column has rows in
# both 2-letter ("NY") and full-name ("NEW YORK") forms; matching only one
# silently drops the other half. Extend this map as new states show up in
# practice; unknown inputs fall back to a single-form exact match.
_STATE_ALIAS_MAP = {
    "NY": ("NY", "NEW YORK"),
    "NEW YORK": ("NY", "NEW YORK"),
    "PA": ("PA", "PENNSYLVANIA"),
    "PENNSYLVANIA": ("PA", "PENNSYLVANIA"),
    "OH": ("OH", "OHIO"),
    "OHIO": ("OH", "OHIO"),
    "NJ": ("NJ", "NEW JERSEY"),
    "NEW JERSEY": ("NJ", "NEW JERSEY"),
    "CA": ("CA", "CALIFORNIA"),
    "CALIFORNIA": ("CA", "CALIFORNIA"),
    "FL": ("FL", "FLORIDA"),
    "FLORIDA": ("FL", "FLORIDA"),
    "TX": ("TX", "TEXAS"),
    "TEXAS": ("TX", "TEXAS"),
}


def _state_aliases(state_input: str) -> tuple[str, ...]:
    key = state_input.strip().upper()
    return _STATE_ALIAS_MAP.get(key, (key,))


def _has_diagnosis_criterion(criteria) -> bool:
    """True when a diagnosis filter is active: codes and/or an active date window.
    An all-None diagnosis_date_range carries no filter and does not count.
    """
    _dr = getattr(criteria, "diagnosis_date_range", None)
    _dr_active = _dr is not None and (getattr(_dr, "start", None) or getattr(_dr, "end", None))
    return bool(getattr(criteria, "diagnosis_codes", None) or _dr_active)


def _diagnosis_event_where(criteria) -> tuple[list[str], dict] | None:
    """Build the WHERE fragment for the diagnosis-event subquery.

    Returns (filter_clauses, params) when a diagnosis criterion is active
    (codes and/or an active date window), else None. Shared by cohort_sql
    (DISTINCT patient_id join) and the breakdown builder (event-level join
    that preserves start_date). Param keys: dx_<i>, dx_start, dx_end.
    """
    _dr = getattr(criteria, "diagnosis_date_range", None)
    _dr_active = _dr is not None and (getattr(_dr, "start", None) or getattr(_dr, "end", None))
    if not _has_diagnosis_criterion(criteria):
        return None

    params: dict = {}
    dx_filter = ["code_type IN ('ICD-10', 'ICD10', 'SNOMED')"]
    if getattr(criteria, "diagnosis_codes", None):
        codes = list(criteria.diagnosis_codes)
        placeholders = ", ".join(f":dx_{i}" for i in range(len(codes)))
        for i, c in enumerate(codes):
            params[f"dx_{i}"] = c
        dx_filter.insert(0, f"code IN ({placeholders})")
    if _dr_active:
        if getattr(_dr, "start", None):
            dx_filter.append("start_date >= :dx_start")
            params["dx_start"] = _dr.start.isoformat()
        if getattr(_dr, "end", None):
            dx_filter.append("start_date <= :dx_end")
            params["dx_end"] = _dr.end.isoformat()
    return dx_filter, params


def _build_non_diagnosis_filters(criteria, schema_prefix: str) -> tuple[list[str], list[str], dict]:
    """Build (join_clauses, where_clauses, params) for everything except
    the diagnosis filter. Shared by cohort_sql and the breakdown builder so
    geo/demographic/medication logic lives in exactly one place.
    """
    params: dict = {}
    join_clauses: list[str] = []
    where_clauses: list[str] = []

    if getattr(criteria, "medication_rxnorm_codes", None):
        codes = list(criteria.medication_rxnorm_codes)
        placeholders = ", ".join(f":med_{i}" for i in range(len(codes)))
        for i, c in enumerate(codes):
            params[f"med_{i}"] = c
        join_clauses.append(
            f"JOIN (SELECT DISTINCT patient_id FROM {schema_prefix}metric_federated_data_v "
            f"WHERE code IN ({placeholders}) "
            f"AND code_type IN ('RxNorm', 'NDC')) med ON med.patient_id = p.patient_id"
        )

    if getattr(criteria, "zip_code", None):
        where_clauses.append("p.zip_code = :geo_zip")
        params["geo_zip"] = criteria.zip_code
    if getattr(criteria, "city", None):
        where_clauses.append("LOWER(p.city) LIKE :geo_city")
        params["geo_city"] = f"%{criteria.city.lower()}%"
    if getattr(criteria, "state", None):
        forms = _state_aliases(criteria.state)
        placeholders = ", ".join(f":geo_state_{i}" for i in range(len(forms)))
        where_clauses.append(f"UPPER(p.state) IN ({placeholders})")
        for i, form in enumerate(forms):
            params[f"geo_state_{i}"] = form

    if getattr(criteria, "age_min", None) is not None:
        where_clauses.append("p.age >= :age_min")
        params["age_min"] = criteria.age_min
    if getattr(criteria, "age_max", None) is not None:
        where_clauses.append("p.age <= :age_max")
        params["age_max"] = criteria.age_max
    if getattr(criteria, "gender", None):
        where_clauses.append("p.gender = :gender")
        params["gender"] = criteria.gender
    if getattr(criteria, "facility", None):
        where_clauses.append("LOWER(p.practice_name) LIKE :facility")
        params["facility"] = f"%{criteria.facility.lower()}%"
    if getattr(criteria, "last_visit_after", None):
        where_clauses.append("p.last_date_of_visit >= :lv_after")
        params["lv_after"] = criteria.last_visit_after.isoformat()
    if getattr(criteria, "last_visit_before", None):
        where_clauses.append("p.last_date_of_visit <= :lv_before")
        params["lv_before"] = criteria.last_visit_before.isoformat()
    if getattr(criteria, "condition_text", None):
        where_clauses.append("LOWER(p.conditions) LIKE :cond_text")
        params["cond_text"] = f"%{criteria.condition_text.lower()}%"

    return join_clauses, where_clauses, params


def cohort_sql(criteria, schema_prefix: str = "") -> Tuple[str, dict]:
    """Build the SELECT for search_patients_by_criteria.

    `criteria` is a CohortCriteria (defined in agent/tools/search_patients_by_criteria.py).
    It is duck-typed here so the SQL builder can be unit-tested against a
    minimal namespace without importing the tool module.

    Returns (sql, params). The caller passes both into adapter.fetch_all.
    """
    if not any(
        (
            _has_diagnosis_criterion(criteria),
            getattr(criteria, "medication_rxnorm_codes", None),
            getattr(criteria, "condition_text", None),
            getattr(criteria, "age_min", None) is not None,
            getattr(criteria, "age_max", None) is not None,
            getattr(criteria, "gender", None),
            getattr(criteria, "facility", None),
            getattr(criteria, "last_visit_after", None),
            getattr(criteria, "last_visit_before", None),
            getattr(criteria, "zip_code", None),
            getattr(criteria, "city", None),
            getattr(criteria, "state", None),
            getattr(criteria, "inpatient_admission", None),
        )
    ):
        raise ValueError("cohort_sql requires at least one search criterion; refusing to issue an unfiltered scan.")

    params: dict = {}
    join_clauses: list[str] = []
    # 1=1 is a structural placeholder so the WHERE always parses when
    # only JOIN-based filters (codes) are active.
    where_clauses: list[str] = ["1=1"]

    dx = _diagnosis_event_where(criteria)
    if dx is not None:
        dx_filter, dx_params = dx
        params.update(dx_params)
        join_clauses.append(
            f"JOIN (SELECT DISTINCT patient_id FROM {schema_prefix}metric_federated_data_v "
            f"WHERE {' AND '.join(dx_filter)}) dx ON dx.patient_id = p.patient_id"
        )

    other_joins, other_where, other_params = _build_non_diagnosis_filters(criteria, schema_prefix)
    join_clauses.extend(other_joins)
    where_clauses.extend(other_where)
    params.update(other_params)

    sample_size = getattr(criteria, "sample_size", 20)
    join_block = "\n        ".join(join_clauses)
    where_block = " AND ".join(where_clauses)

    if int(sample_size) == 0:
        # Count-only fast path. The COUNT(*) OVER () + LIMIT shape that
        # the per-row query uses still forces Postgres to compute the
        # window over the full result before applying the LIMIT, which
        # for broad cohorts (e.g. all-NY + 3 diagnosis codes) can run
        # for many minutes. A pure aggregate is cheap regardless of
        # population size.
        #
        # Count distinct source_id (the EMPI-resolved canonical person
        # identifier). isr.empi_rank = 1 selects each person's current
        # primary CID (one per person); counting source_id at rank 1 =
        # distinct people. empi_rank != 99 would include legacy merged CIDs
        # and over-count ~2x. Using patient_id here would over-count people
        # who have multiple internal patient_ids. This matches the sample
        # path's COUNT(*) OVER () grain and the breakdown path's
        # COUNT(DISTINCT source_id).
        sql = f"""
            SELECT COUNT(DISTINCT isr.source_id) AS total_count
            FROM {schema_prefix}internal_patient_profile_v p
            JOIN {schema_prefix}internal_source_reference_v isr
              ON isr.patient_id = p.patient_id
              AND isr.empi_rank = 1
            {join_block}
            WHERE {where_block}
        """
        return sql, params

    params["sample_size_plus_one"] = int(sample_size) + 1
    sql = f"""
        SELECT
            isr.source_id,
            p.full_name AS display_name,
            p.age,
            p.gender,
            p.last_date_of_visit,
            p.practice_name,
            COUNT(*) OVER () AS total_count
        FROM {schema_prefix}internal_patient_profile_v p
        JOIN {schema_prefix}internal_source_reference_v isr
          ON isr.patient_id = p.patient_id
          AND isr.empi_rank = 1
        {join_block}
        WHERE {where_block}
        LIMIT :sample_size_plus_one
    """
    return sql, params
