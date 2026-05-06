"""SQL templates for clinical data domains.

Each domain has its own function returning (sql, params). Phase 1
implements demographics + encounters; Phase 2 adds the rest.

Per spec §7.12: queries hit the federated_*_v whitelist. Per §8.2:
all clinical filters key on source_id.
"""

from __future__ import annotations
from typing import Optional, Tuple


_FACILITY_TYPE_TO_POS_CODES = {
    # CMS Place of Service codes — see https://www.cms.gov/medicare/coding-billing/place-of-service-codes
    "inpatient": ("21",),
    "outpatient": ("22",),
    "ed": ("23",),
    "ltc": ("32", "33", "34", "54", "56"),
}


def demographics_sql(*, source_id: str) -> Tuple[str, dict]:
    return (
        """
        SELECT
            source_id,
            first_name,
            last_name,
            date_of_birth,
            gender
        FROM federated_demographic_v
        WHERE source_id = :source_id
        """,
        {"source_id": source_id},
    )


def encounters_sql(
    *,
    source_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    facility_type: Optional[str] = None,
) -> Tuple[str, dict]:
    where_clauses = ["source_id = :source_id"]
    params: dict = {"source_id": source_id}

    # Per §7.12: prefer status_datetime when available; fall back to datetime.
    date_col = "COALESCE(status_datetime, datetime)"
    if start_date:
        where_clauses.append(f"{date_col} >= :start_date")
        params["start_date"] = start_date
    if end_date:
        where_clauses.append(f"{date_col} <= :end_date")
        params["end_date"] = end_date

    if facility_type and facility_type != "any":
        codes = _FACILITY_TYPE_TO_POS_CODES.get(facility_type, ())
        if codes:
            placeholders = ", ".join(f":pos_{i}" for i in range(len(codes)))
            where_clauses.append(f"place_of_service IN ({placeholders})")
            for i, c in enumerate(codes):
                params[f"pos_{i}"] = c

    where = " AND ".join(where_clauses)
    sql = f"""
        SELECT
            source_id,
            encounter_id,
            type,
            status,
            {date_col} AS event_datetime,
            location,
            rendering_provider,
            facility_name,
            place_of_service
        FROM federated_encounters_v
        WHERE {where}
        ORDER BY {date_col} DESC
    """
    return sql, params
