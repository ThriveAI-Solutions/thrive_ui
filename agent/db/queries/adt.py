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


# --- Inpatient-admission predicate (single source of truth) ---
# See docs/superpowers/specs/2026-06-26-adt-inpatient-admission-design.md.
INPATIENT_SETTINGS = ("INPATIENT",)
OP_TO_IP_STATUS = "A06"  # outpatient -> inpatient conversion (clean inpatient signal)
PREADMIT_PENDING_STATUSES = ("A05", "A14", "A38", "A27")  # not an admission even w/ IP setting
CANCEL_ADMIT_STATUS = "CANCEL ADMIT"  # voids the whole visit


def _sql_str_list(values: tuple[str, ...]) -> str:
    return ", ".join("'" + v.replace("'", "''") + "'" for v in values)


def _qualifying_evidence_sql(alias: str = "adt") -> str:
    """Row-level predicate: this row is positive evidence of an inpatient stay.

    Inpatient setting OR an A06 conversion; not a pre-admit/pending status;
    not a cancelled row. COALESCE guards NULLs so valid rows don't drop out
    via SQL three-valued logic.
    """
    return (
        f"(UPPER({alias}.clean_setting) IN ({_sql_str_list(INPATIENT_SETTINGS)}) "
        f"OR {alias}.clean_status = '{OP_TO_IP_STATUS}') "
        f"AND COALESCE({alias}.clean_status, '') NOT IN ({_sql_str_list(PREADMIT_PENDING_STATUSES)}) "
        f"AND COALESCE({alias}.cancelled_flag, 'N') <> 'Y'"
    )


def inpatient_admission_flag_sql(dialect: str, alias: str = "adt") -> str:
    """Visit-level boolean: does this visit_number represent an inpatient admission?

    Use inside a query that GROUPs BY visit_number (SELECT column or HAVING).
    """
    evidence = _qualifying_evidence_sql(alias)
    cancel = f"{alias}.clean_status = '{CANCEL_ADMIT_STATUS}'"
    if dialect in ("postgres", "redshift"):
        return f"BOOL_OR({evidence}) AND NOT BOOL_OR({cancel})"
    if dialect == "sqlite":
        return f"MAX(CASE WHEN {evidence} THEN 1 ELSE 0 END) = 1 AND MAX(CASE WHEN {cancel} THEN 1 ELSE 0 END) = 0"
    raise ValueError(f"Unsupported dialect for ADT predicate: {dialect!r}")


def qualifying_admit_date_sql(alias: str = "adt") -> str:
    """Visit-level admission date = earliest event_date among qualifying rows."""
    return f"MIN(CASE WHEN {_qualifying_evidence_sql(alias)} THEN {alias}.event_date END)"


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
