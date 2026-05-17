"""Aggregate clinical + claims rows into dw.metric_federated_data_v."""

from __future__ import annotations

import datetime as dt

from scripts.sample_db.transformers.base import TransformContext


def transform_rollup(
    source_to_patient: dict[str, int],
    source_to_practice: dict[str, str],
    ctx: TransformContext,
    *,
    claim_to_source: dict[str, str] | None = None,
) -> None:
    claim_to_source = claim_to_source or {}
    out = []

    # Problems → clinical rollup.
    for r in ctx.output.get("dw.federated_problems_v", []):
        pid = source_to_patient.get(r["source_id"])
        if pid is None:
            continue
        out.append(
            {
                "patient_id": pid,
                "origin_id": r.get("source_id"),
                "start_date": _date(r.get("diagnosis_datetime")),
                "end_date": None,
                "code": r.get("code"),
                "code_type": r.get("code_type"),
                "code_system": _system(r.get("code_type")),
                "event_name": r.get("diagnosis"),
                "source_table": "federated_problems_v",
                "is_claims_data": 0,
                "data_source": source_to_practice.get(r["source_id"], "Unknown"),
            }
        )

    # Meds → clinical rollup.
    for r in ctx.output.get("dw.federated_meds_v", []):
        pid = source_to_patient.get(r["source_id"])
        if pid is None:
            continue
        out.append(
            {
                "patient_id": pid,
                "origin_id": r.get("source_id"),
                "start_date": _date(r.get("date_prescribed")),
                "end_date": None,
                "code": r.get("rxnorm_code"),
                "code_type": "RxNorm",
                "code_system": "RXNORM",
                "event_name": r.get("med_name"),
                "source_table": "federated_meds_v",
                "is_claims_data": 0,
                "data_source": source_to_practice.get(r["source_id"], "Unknown"),
            }
        )

    # Claims diagnoses → claims rollup.
    for r in ctx.output.get("dw.federated_claims_icd_diagnosis_detail_v", []):
        line_id = r.get("claim_line_identifier")
        sid = claim_to_source.get(line_id)
        if sid is None:
            continue
        pid = source_to_patient.get(sid)
        if pid is None:
            continue
        out.append(
            {
                "patient_id": pid,
                "origin_id": line_id,
                "start_date": _date(r.get("diagnosis_date")),
                "end_date": None,
                "code": r.get("icd_diagnosis_code"),
                "code_type": r.get("icd_type"),
                "code_system": "ICD10CM",
                "event_name": None,
                "source_table": "federated_claims_icd_diagnosis_detail_v",
                "is_claims_data": 1,
                "data_source": source_to_practice.get(sid, "Claims"),
            }
        )

    ctx.add_rows("dw.metric_federated_data_v", out)


def _date(value):
    if value is None:
        return None
    if isinstance(value, (dt.date, dt.datetime)):
        return value.date() if isinstance(value, dt.datetime) else value
    return dt.date.fromisoformat(str(value)[:10])


def _system(code_type: str | None) -> str | None:
    if not code_type:
        return None
    s = code_type.upper().replace("-", "").replace(" ", "")
    if s in {"ICD10", "ICD10CM"}:
        return "ICD10CM"
    if s in {"SNOMED", "SNOMEDCT"}:
        return "SNOMEDCT"
    if s in {"ICD9", "ICD9CM"}:
        return "ICD9CM"
    return code_type
