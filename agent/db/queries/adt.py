"""SQL template for federated_adt_v (admit/discharge/transfer events).

Supports filtering by facility type (inpatient, ltc/snf, ed, outpatient, any),
date range, and discharge detail inclusion. The clean_setting column
normalizes setting values to INPATIENT, OUTPATIENT, EMERGENCY, etc.

Unlike every other federated_*_v view, federated_adt_v exposes only
`patient_id` (no `source_id`). The agent's identity model is source_id-based,
so we resolve source_id → patient_id through internal_source_reference_v
at empi_rank = 1 — same pattern as patient.find_patient_sql.
"""

from __future__ import annotations
from typing import Optional, Tuple


_FACILITY_TYPE_SETTINGS = {
    "inpatient": ("INPATIENT",),
    "ltc": ("LONG TERM CARE", "SNF", "SKILLED NURSING"),
    "snf": ("LONG TERM CARE", "SNF", "SKILLED NURSING"),
    "ed": ("EMERGENCY",),
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
    where: list[str] = ["isr.source_id = :source_id", "isr.empi_rank = 1"]
    params: dict = {"source_id": source_id}

    if facility_type and facility_type != "any":
        settings = _FACILITY_TYPE_SETTINGS.get(facility_type)
        if settings:
            placeholders = ", ".join(f":ft_{i}" for i in range(len(settings)))
            where.append(f"UPPER(adt.clean_setting) IN ({placeholders})")
            for i, s in enumerate(settings):
                params[f"ft_{i}"] = s

    if start_date:
        where.append("adt.event_date >= :start_date")
        params["start_date"] = start_date
    if end_date:
        where.append("adt.event_date <= :end_date")
        params["end_date"] = end_date

    discharge_cols = ""
    if include_discharge_details:
        discharge_cols = """,
            adt.discharge_disposition,
            adt.discharge_location"""

    # `isr.source_id AS source_id` echoes the input identifier into each row
    # so _build_admissions_result can populate AdmissionItem.source_id without
    # special-casing this query.
    sql = f"""
        SELECT
            isr.source_id AS source_id,
            adt.event_date,
            adt.event_location,
            adt.location_type,
            adt.clean_setting AS setting,
            adt.status,
            adt.admit_from{discharge_cols}
        FROM {schema_prefix}federated_adt_v adt
        JOIN {schema_prefix}internal_source_reference_v isr
          ON isr.patient_id = adt.patient_id
        WHERE {" AND ".join(where)}
        ORDER BY adt.event_date DESC
    """
    return sql, params
