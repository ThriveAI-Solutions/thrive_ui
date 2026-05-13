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


def cohort_sql(criteria, schema_prefix: str = "") -> Tuple[str, dict]:
    """Build the SELECT for search_patients_by_criteria.

    `criteria` is a CohortCriteria (defined in agent/tools/search_patients_by_criteria.py).
    It is duck-typed here so the SQL builder can be unit-tested against a
    minimal namespace without importing the tool module.

    Returns (sql, params). The caller passes both into adapter.fetch_all.
    """
    if not any(
        (
            getattr(criteria, "diagnosis_codes", None),
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
        )
    ):
        raise ValueError("cohort_sql requires at least one search criterion; refusing to issue an unfiltered scan.")

    params: dict = {}
    join_clauses: list[str] = []
    # 1=1 is a structural placeholder so the WHERE always parses when
    # only JOIN-based filters (codes) are active. The criterion guard
    # above guarantees this is never an unfiltered scan.
    where_clauses: list[str] = ["1=1"]

    # --- diagnosis codes ----------------------------------------------
    if getattr(criteria, "diagnosis_codes", None):
        codes = list(criteria.diagnosis_codes)
        placeholders = ", ".join(f":dx_{i}" for i in range(len(codes)))
        for i, c in enumerate(codes):
            params[f"dx_{i}"] = c
        dx_filter = [
            f"code IN ({placeholders})",
            "code_type IN ('ICD-10', 'ICD10', 'SNOMED')",
        ]
        dr = getattr(criteria, "diagnosis_date_range", None)
        if dr is not None:
            if getattr(dr, "start", None):
                dx_filter.append("start_date >= :dx_start")
                params["dx_start"] = dr.start.isoformat()
            if getattr(dr, "end", None):
                dx_filter.append("start_date <= :dx_end")
                params["dx_end"] = dr.end.isoformat()
        join_clauses.append(
            f"JOIN (SELECT DISTINCT patient_id FROM {schema_prefix}metric_federated_data_v "
            f"WHERE {' AND '.join(dx_filter)}) dx ON dx.patient_id = p.patient_id"
        )

    # --- medication codes ---------------------------------------------
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

    # --- geographic filters (direct on internal_patient_profile_v) -----
    # Live probe (May 2026) confirmed zip_code/city/state are structured
    # columns on internal_patient_profile_v with 100% zip coverage. No
    # JOIN to federated_demographic_v is needed; that view has the same
    # data but the cohort path already lands on internal_patient_profile_v.
    if getattr(criteria, "zip_code", None):
        where_clauses.append("p.zip_code = :geo_zip")
        params["geo_zip"] = criteria.zip_code
    if getattr(criteria, "city", None):
        # City strings are inconsistent ("BUFFALO" vs "Buffalo" vs "TN OF TONA");
        # case-insensitive substring is the safest match.
        where_clauses.append("LOWER(p.city) LIKE :geo_city")
        params["geo_city"] = f"%{criteria.city.lower()}%"
    if getattr(criteria, "state", None):
        # internal_patient_profile_v.state holds both 2-letter codes
        # ("NY") and full names ("NEW YORK") inconsistently. Match either
        # form when we can map the input to its alias; fall back to a
        # single-form exact match for unknown inputs. Without this,
        # state="NY" silently misses every "NEW YORK" row and vice versa.
        forms = _state_aliases(criteria.state)
        placeholders = ", ".join(f":geo_state_{i}" for i in range(len(forms)))
        where_clauses.append(f"UPPER(p.state) IN ({placeholders})")
        for i, form in enumerate(forms):
            params[f"geo_state_{i}"] = form

    # --- demographic / visit filters ----------------------------------
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

    sample_size = getattr(criteria, "sample_size", 20)
    params["sample_size_plus_one"] = int(sample_size) + 1

    join_block = "\n        ".join(join_clauses)
    where_block = " AND ".join(where_clauses)

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
          AND isr.empi_rank != 99
        {join_block}
        WHERE {where_block}
        LIMIT :sample_size_plus_one
    """
    return sql, params
