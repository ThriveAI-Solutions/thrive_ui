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
PREADMIT_SETTINGS = ("P",)  # HL7 PV1-2 patient class P = preadmit; not a completed encounter
CANCEL_ADMIT_STATUS = "CANCEL ADMIT"  # voids the whole visit
# Lifecycle statuses are PREFERRED for anchoring a visit's start date / admitting
# facility. They are NOT required for a visit to exist: some federated source
# systems (e.g. radiology groups) report a real encounter with only A08/A31.
VISIT_LIFECYCLE_STATUSES = (
    "ADMIT",
    "REGISTRATION",
    "TRANSFER",
    "DISCHARGE",
    "A01",
    "A02",
    "A03",
    "A04",
    "A06",
    "A07",
)
DISCHARGE_STATUSES = ("DISCHARGE", "A03")


def _sql_str_list(values: tuple[str, ...]) -> str:
    return ", ".join("'" + v.replace("'", "''") + "'" for v in values)


def patient_id_text_sql(alias: str, column: str = "patient_id") -> str:
    """Return a text expression for patient_id joins against federated_adt_v.

    Production federated_adt_v.patient_id is VARCHAR while identity/profile
    tables expose integer patient_id. Cast the integer side to text instead
    of casting ADT to numeric, because a dirty ADT value should not make the
    whole query fail.
    """
    return f"CAST({alias}.{column} AS VARCHAR)"


def visit_number_sql(alias: str) -> str:
    """Normalized ADT visit number, blank strings treated as missing."""
    return f"NULLIF(TRIM(CAST({alias}.visit_number AS VARCHAR)), '')"


def visit_key_sql(alias: str) -> str:
    """Stable grouping key for ADT visits.

    Most rows have visit_number, but production has millions of ADT events
    with NULL/blank visit_number. Falling back to a per-event descriptive key
    avoids collapsing every missing-visit row for a patient into one fake stay.
    """
    return (
        f"COALESCE({visit_number_sql(alias)}, "
        f"'__missing_visit__:' || COALESCE(CAST({alias}.event_date AS VARCHAR), '') || "
        f"':' || COALESCE({alias}.clean_status, '') || "
        f"':' || COALESCE({alias}.event_location, '') || "
        f"':' || COALESCE({alias}.location_type, '') || "
        f"':' || COALESCE({alias}.clean_setting, ''))"
    )


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


def _clean_status_in_sql(alias: str, values: tuple[str, ...]) -> str:
    return f"UPPER(COALESCE({alias}.clean_status, '')) IN ({_sql_str_list(values)})"


def _not_preadmit_sql(alias: str = "adt") -> str:
    """Row is not a pre-admit/pending placeholder.

    Excludes both the pre-admit/pending *statuses* (A05/A14/A38/A27) and the
    HL7 PV1-2 pre-admit patient-class *setting* ('P'). A pre-admit is a
    scheduled-but-not-yet-occurred encounter, never a completed visit.
    """
    return (
        f"UPPER(COALESCE({alias}.clean_status, '')) NOT IN ({_sql_str_list(PREADMIT_PENDING_STATUSES)}) "
        f"AND UPPER(COALESCE({alias}.clean_setting, '')) NOT IN ({_sql_str_list(PREADMIT_SETTINGS)})"
    )


def _visit_event_sql(alias: str = "adt") -> str:
    """Row-level predicate: this row reflects a real patient-facility contact.

    In this federated feed some source systems (notably radiology groups and
    some practices) report a genuine encounter using ONLY administrative HL7
    messages (A08 update / A31 update-person), so a lifecycle status is NOT
    required. Any row that is not a pre-admit/pending placeholder counts as a
    real contact. Cancellation is handled at the visit level (a CANCEL ADMIT
    voids the whole visit), not here, because it is visit-scoped: the cancel
    message nullifies a different (admit) row that is itself never flagged.
    """
    return _not_preadmit_sql(alias)


def _lifecycle_event_sql(alias: str = "adt") -> str:
    """Row-level predicate: this row is a visit *lifecycle* event
    (registration / admit / transfer / discharge).

    Used only to PREFER a real lifecycle event over an administrative update
    when anchoring a visit's start date and admitting facility; not used to
    decide whether the visit exists (see :func:`_visit_event_sql`).
    """
    return f"{_clean_status_in_sql(alias, VISIT_LIFECYCLE_STATUSES)} AND {_not_preadmit_sql(alias)}"


def inpatient_admission_flag_sql(dialect: str, alias: str = "adt") -> str:
    """Visit-level boolean: does this visit_number represent an inpatient admission?

    Use inside a query that GROUPs BY visit_key_sql(alias) (SELECT column or HAVING).
    """
    evidence = _qualifying_evidence_sql(alias)
    cancel = f"COALESCE({alias}.clean_status, '') = '{CANCEL_ADMIT_STATUS}'"
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
    patient_id_expr = patient_id_text_sql("adt")
    visit_key_expr = visit_key_sql("adt")
    having = [flag]
    if start_date:
        having.append(f"{qad} >= :{param_prefix}_start")
        params[f"{param_prefix}_start"] = start_date
    if end_date:
        having.append(f"{qad} <= :{param_prefix}_end")
        params[f"{param_prefix}_end"] = end_date

    if project_admit_date:
        select = f"{patient_id_expr} AS patient_id, {qad} AS qualifying_admit_date"
    else:
        select = f"DISTINCT {patient_id_expr} AS patient_id"

    sql = (
        f"SELECT {select}\n"
        f"        FROM {schema_prefix}federated_adt_v adt\n"
        f"        GROUP BY {patient_id_expr}, {visit_key_expr}\n"
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
    """One row per visit for a patient, with a computed is_inpatient_admission
    flag (0/1). See the design doc for the definition.

    facility_type='inpatient' filters to is_inpatient_admission stays and
    date-filters on the derived qualifying_admit_date; other facility types
    filter to stays containing that clean_setting and date-filter on admit_date;
    'any'/None returns all stays. Rows with missing/blank visit_number use a
    per-event synthetic grouping key so they do not collapse into one fake
    stay. The CTE rolls up the WHOLE keyed visit so an out-of-window CANCEL
    ADMIT can still void it when visit_number is present.
    """
    params: dict = {"source_id": source_id}
    flag_expr = inpatient_admission_flag_sql(dialect, alias="adt")
    qad_expr = qualifying_admit_date_sql(alias="adt")
    evidence_expr = _qualifying_evidence_sql(alias="adt")
    visit_event_expr = _visit_event_sql(alias="adt")
    lifecycle_expr = _lifecycle_event_sql(alias="adt")
    cancel_expr = f"COALESCE(adt.clean_status, '') = '{CANCEL_ADMIT_STATUS}'"
    discharge_expr = _clean_status_in_sql("adt", DISCHARGE_STATUSES)
    visit_key_expr = visit_key_sql("adt")

    facility_has_col = ""
    if facility_type and facility_type not in ("any", "inpatient"):
        settings = _FACILITY_TYPE_SETTINGS.get(facility_type)
        if settings:
            facility_has_col = (
                f",\n                MAX(CASE WHEN UPPER(adt.clean_setting) IN "
                f"({_sql_str_list(settings)}) AND adt.is_visit_event = 1 "
                f"THEN 1 ELSE 0 END) AS has_facility"
            )

    outer_where: list[str] = ["1=1"]
    if facility_type == "inpatient":
        outer_where.append("is_inpatient_admission = 1")
        date_col = "qualifying_admit_date"
    elif facility_type and facility_type != "any" and _FACILITY_TYPE_SETTINGS.get(facility_type):
        outer_where.append("has_facility = 1")
        date_col = "COALESCE(visit_start_date, admit_date)"
    else:
        outer_where.append("(has_visit_event = 1 OR is_inpatient_admission = 1)")
        date_col = (
            "CASE WHEN is_inpatient_admission = 1 AND qualifying_admit_date IS NOT NULL "
            "THEN qualifying_admit_date ELSE COALESCE(visit_start_date, admit_date) END"
        )

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
        f"                    PARTITION BY isr.source_id, {visit_key_expr}\n"
        f"                    ORDER BY CASE WHEN {evidence_expr} THEN 0 WHEN {lifecycle_expr} THEN 1 "
        f"WHEN {visit_event_expr} THEN 2 ELSE 3 END, adt.event_date\n"
        f"                )"
    )
    display_admit_date_expr = (
        "CASE\n"
        "                WHEN is_inpatient_admission = 1 AND qualifying_admit_date IS NOT NULL\n"
        "                    THEN qualifying_admit_date\n"
        "                ELSE COALESCE(visit_start_date, admit_date)\n"
        "            END"
    )

    sql = f"""
        WITH ranked AS (
            SELECT
                isr.source_id AS source_id,
                {visit_number_sql("adt")} AS visit_number,
                {visit_key_expr} AS visit_key,
                adt.event_date AS event_date,
                adt.clean_status AS clean_status,
                adt.clean_setting AS clean_setting,
                adt.cancelled_flag AS cancelled_flag,
                adt.event_location AS event_location,
                adt.location_type AS location_type,
                adt.admit_from AS admit_from,
                adt.discharge_disposition AS discharge_disposition,
                adt.discharge_location AS discharge_location,
                CASE WHEN {visit_event_expr} THEN 1 ELSE 0 END AS is_visit_event,
                CASE WHEN {lifecycle_expr} THEN 1 ELSE 0 END AS is_lifecycle_event,
                CASE WHEN {cancel_expr} THEN 1 ELSE 0 END AS is_cancel_admit,
                {admit_rn_expr} AS admit_rn
            FROM {schema_prefix}federated_adt_v adt
            JOIN {schema_prefix}internal_source_reference_v isr
              ON {patient_id_text_sql("isr")} = adt.patient_id AND isr.empi_rank = 1
            WHERE isr.source_id = :source_id
        ),
        visit_rollup AS (
            SELECT
                source_id AS source_id,
                MAX(adt.visit_number) AS visit_number,
                MIN(adt.event_date) AS admit_date,
                COALESCE(
                    MIN(CASE WHEN adt.is_lifecycle_event = 1 THEN adt.event_date END),
                    MIN(CASE WHEN adt.is_visit_event = 1 THEN adt.event_date END)
                ) AS visit_start_date,
                {qad_expr} AS qualifying_admit_date,
                MAX(CASE WHEN {discharge_expr} THEN adt.event_date END) AS discharge_date,
                CASE WHEN ({flag_expr}) THEN 1 ELSE 0 END AS is_inpatient_admission,
                CASE WHEN MAX(adt.is_visit_event) = 1 AND MAX(adt.is_cancel_admit) = 0
                     THEN 1 ELSE 0 END AS has_visit_event,
                COALESCE(
                    MAX(CASE WHEN {evidence_expr} THEN NULLIF(adt.clean_setting, '') END),
                    MAX(CASE WHEN adt.is_visit_event = 1 THEN NULLIF(adt.clean_setting, '') END),
                    MAX(NULLIF(adt.clean_setting, ''))
                ) AS setting,
                MAX(CASE WHEN adt.admit_rn = 1 THEN adt.event_location END) AS event_location,
                MAX(CASE WHEN adt.admit_rn = 1 THEN adt.location_type END) AS location_type,
                MAX(adt.admit_from) AS admit_from,
                MAX(CASE WHEN {discharge_expr} THEN adt.discharge_disposition END) AS discharge_disposition,
                MAX(CASE WHEN {discharge_expr} THEN adt.discharge_location END) AS discharge_location
                {facility_has_col}
            FROM ranked adt
            GROUP BY source_id, adt.visit_key
        )
        SELECT
            source_id,
            visit_number,
            {display_admit_date_expr} AS admit_date,
            discharge_date,
            setting,
            is_inpatient_admission, event_location, location_type, admit_from,
            discharge_disposition, discharge_location
        FROM visit_rollup
        WHERE {" AND ".join(outer_where)}
        ORDER BY admit_date DESC
    """
    return sql, params
