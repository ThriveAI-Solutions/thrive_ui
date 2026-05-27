"""SQL template for federated_adt_v (admit/discharge/transfer events).

Supports filtering by facility type (inpatient, ltc/snf, emergency, any),
date range, and discharge detail inclusion. The clean_setting column
normalizes setting values to INPATIENT, OUTPATIENT, EMERGENCY, etc.
"""

from __future__ import annotations
from typing import Optional, Tuple


_FACILITY_TYPE_SETTINGS = {
    "inpatient": ("INPATIENT",),
    "ltc": ("LONG TERM CARE", "SNF", "SKILLED NURSING"),
    "snf": ("LONG TERM CARE", "SNF", "SKILLED NURSING"),
    "emergency": ("EMERGENCY",),
    "outpatient": ("OUTPATIENT",),
}


def admissions_sql(
    *,
    source_id: str,
    facility_type: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    include_discharge_details: bool = True,
    schema_prefix: str = "",
) -> Tuple[str, dict]:
    where: list[str] = ["source_id = :source_id"]
    params: dict = {"source_id": source_id}

    if facility_type and facility_type != "any":
        settings = _FACILITY_TYPE_SETTINGS.get(facility_type)
        if settings:
            placeholders = ", ".join(f":ft_{i}" for i in range(len(settings)))
            where.append(f"UPPER(clean_setting) IN ({placeholders})")
            for i, s in enumerate(settings):
                params[f"ft_{i}"] = s

    if start_date:
        where.append("event_date >= :start_date")
        params["start_date"] = start_date
    if end_date:
        where.append("event_date <= :end_date")
        params["end_date"] = end_date

    discharge_cols = ""
    if include_discharge_details:
        discharge_cols = """,
            discharge_disposition,
            discharge_location"""

    sql = f"""
        SELECT
            source_id,
            event_date,
            event_location,
            location_type,
            clean_setting AS setting,
            status,
            admit_from{discharge_cols}
        FROM {schema_prefix}federated_adt_v
        WHERE {" AND ".join(where)}
        ORDER BY event_date DESC
    """
    return sql, params
