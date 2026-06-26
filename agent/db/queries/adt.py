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


def inpatient_cohort_subquery_sql(
    dialect: str,
    schema_prefix: str = "",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    project_admit_date: bool = False,
    param_prefix: str = "adt",
) -> Tuple[str, dict]:
    """Visit-grouped ADT subquery for cohort use.

    project_admit_date=False → SELECT DISTINCT patient_id (filter; dedup so the
    cohort COUNT(*) OVER () is not inflated by multiple stays per patient).
    project_admit_date=True → one row per qualifying visit with qualifying_admit_date
    (breakdown bucketing).
    """
    params: dict = {}
    flag = inpatient_admission_flag_sql(dialect, alias="adt")
    qad = qualifying_admit_date_sql(alias="adt")
    having = [flag]
    if start_date:
        having.append(f"{qad} >= :{param_prefix}_start")
        params[f"{param_prefix}_start"] = start_date
    if end_date:
        having.append(f"{qad} <= :{param_prefix}_end")
        params[f"{param_prefix}_end"] = end_date

    if project_admit_date:
        select = f"adt.patient_id, {qad} AS qualifying_admit_date"
    else:
        select = "DISTINCT patient_id"

    sql = (
        f"SELECT {select}\n"
        f"        FROM {schema_prefix}federated_adt_v adt\n"
        f"        GROUP BY adt.patient_id, adt.visit_number\n"
        f"        HAVING {' AND '.join(having)}"
    )
    return sql, params


def admissions_sql(
    *,
    source_id: str,
    dialect: str,
    facility_type: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    schema_prefix: str = "",
) -> Tuple[str, dict]:
    """One row per visit_number for a patient, with a computed
    is_inpatient_admission flag (0/1). See the design doc for the definition.

    facility_type='inpatient' filters to is_inpatient_admission stays and
    date-filters on the derived qualifying_admit_date; other facility types
    filter to stays containing that clean_setting and date-filter on admit_date;
    'any'/None returns all stays. The CTE rolls up the WHOLE visit so an
    out-of-window CANCEL ADMIT can still void it.
    """
    params: dict = {"source_id": source_id}
    flag_expr = inpatient_admission_flag_sql(dialect, alias="adt")
    qad_expr = qualifying_admit_date_sql(alias="adt")
    evidence_expr = _qualifying_evidence_sql(alias="adt")

    facility_has_col = ""
    if facility_type and facility_type not in ("any", "inpatient"):
        settings = _FACILITY_TYPE_SETTINGS.get(facility_type)
        if settings:
            facility_has_col = (
                f",\n                MAX(CASE WHEN UPPER(adt.clean_setting) IN "
                f"({_sql_str_list(settings)}) THEN 1 ELSE 0 END) AS has_facility"
            )

    outer_where: list[str] = ["1=1"]
    if facility_type == "inpatient":
        outer_where.append("is_inpatient_admission = 1")
        date_col = "qualifying_admit_date"
    elif facility_type and facility_type != "any" and _FACILITY_TYPE_SETTINGS.get(facility_type):
        outer_where.append("has_facility = 1")
        date_col = "admit_date"
    else:
        date_col = "admit_date"

    if start_date:
        outer_where.append(f"{date_col} >= :start_date")
        params["start_date"] = start_date
    if end_date:
        outer_where.append(f"{date_col} <= :end_date")
        params["end_date"] = end_date

    # admit_rn=1 marks the admitting event of the visit: the earliest qualifying
    # (inpatient/A06) row, or — when the visit has no qualifying row — the
    # earliest event. event_location/location_type are read from that row so a
    # cross-facility transfer (ED at one site -> inpatient at another) reports the
    # admitting facility, not an arbitrary MAX across the visit.
    admit_rn_expr = (
        f"ROW_NUMBER() OVER (\n"
        f"                    PARTITION BY isr.source_id, adt.visit_number\n"
        f"                    ORDER BY CASE WHEN {evidence_expr} THEN 0 ELSE 1 END, adt.event_date\n"
        f"                )"
    )

    sql = f"""
        WITH ranked AS (
            SELECT
                isr.source_id AS source_id,
                adt.visit_number AS visit_number,
                adt.event_date AS event_date,
                adt.clean_status AS clean_status,
                adt.clean_setting AS clean_setting,
                adt.cancelled_flag AS cancelled_flag,
                adt.event_location AS event_location,
                adt.location_type AS location_type,
                adt.admit_from AS admit_from,
                adt.discharge_disposition AS discharge_disposition,
                adt.discharge_location AS discharge_location,
                {admit_rn_expr} AS admit_rn
            FROM {schema_prefix}federated_adt_v adt
            JOIN {schema_prefix}internal_source_reference_v isr
              ON isr.patient_id = adt.patient_id AND isr.empi_rank = 1
            WHERE isr.source_id = :source_id
        ),
        visit_rollup AS (
            SELECT
                source_id AS source_id,
                adt.visit_number AS visit_number,
                MIN(adt.event_date) AS admit_date,
                {qad_expr} AS qualifying_admit_date,
                MAX(CASE WHEN adt.clean_status = 'DISCHARGE' THEN adt.event_date END) AS discharge_date,
                CASE WHEN ({flag_expr}) THEN 1 ELSE 0 END AS is_inpatient_admission,
                COALESCE(
                    MAX(CASE WHEN {evidence_expr} THEN adt.clean_setting END),
                    MAX(adt.clean_setting)
                ) AS setting,
                MAX(CASE WHEN adt.admit_rn = 1 THEN adt.event_location END) AS event_location,
                MAX(CASE WHEN adt.admit_rn = 1 THEN adt.location_type END) AS location_type,
                MAX(adt.admit_from) AS admit_from,
                MAX(CASE WHEN adt.clean_status = 'DISCHARGE' THEN adt.discharge_disposition END) AS discharge_disposition,
                MAX(CASE WHEN adt.clean_status = 'DISCHARGE' THEN adt.discharge_location END) AS discharge_location
                {facility_has_col}
            FROM ranked adt
            GROUP BY source_id, adt.visit_number
        )
        SELECT
            source_id, visit_number, admit_date, discharge_date, setting,
            is_inpatient_admission, event_location, location_type, admit_from,
            discharge_disposition, discharge_location
        FROM visit_rollup
        WHERE {" AND ".join(outer_where)}
        ORDER BY admit_date DESC
    """
    return sql, params
