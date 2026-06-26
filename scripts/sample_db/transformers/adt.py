"""Synthea encounters.csv → dw.federated_adt_v.

Per epic #173: the federated_adt_v view holds admit/discharge/transfer
events sourced from the warehouse's ADT feed. Synthea doesn't model ADT
directly, so we derive ADT events from inpatient + emergency encounters:
- One ADMIT row per inpatient/emergency encounter (other ENCOUNTERCLASS
  values are dropped — outpatient visits aren't admissions).
- When STOP is present, one DISCHARGE row at STOP with discharge details.
  A missing STOP means the encounter is still in-flight, so no discharge
  event/details are emitted.
- Unlike every other federated_*_v table, federated_adt_v exposes only
  patient_id (VARCHAR in production) — the agent's source_id-based identity is
  resolved by joining internal_source_reference_v at empi_rank = 1.
"""

from __future__ import annotations

import pandas as pd

from scripts.sample_db.transformers.base import TransformContext, naive_dt


_SETTING_MAP = {
    "inpatient": ("INPATIENT", "Hospital"),
    "emergency": ("EMERGENCY", "Emergency"),
}


def transform_adt(
    encounters: pd.DataFrame,
    synthea_to_pid: dict[str, int],
    ctx: TransformContext,
) -> None:
    out: list[dict] = []
    for _, e in encounters.iterrows():
        enc_class = str(e.get("ENCOUNTERCLASS", "")).lower()
        if enc_class not in _SETTING_MAP:
            continue
        patient_id = synthea_to_pid.get(e["PATIENT"])
        if patient_id is None:
            continue
        clean_setting, location_type = _SETTING_MAP[enc_class]
        start = naive_dt(e["START"]) if pd.notna(e.get("START")) else None
        stop = naive_dt(e["STOP"]) if pd.notna(e.get("STOP")) else None
        visit_number = f"V{e.get('Id', e.get('ENCOUNTER', patient_id))}"
        base = {
            "patient_id": patient_id,
            "visit_number": visit_number,
            "event_location": str(e.get("ORGANIZATION", "")) or None,
            "location_type": location_type,
            "clean_setting": clean_setting,
            "cancelled_flag": "N",
        }
        out.append(
            {
                **base,
                "event_date": start,
                "status": "Admitted",
                "clean_status": "ADMIT",
                "admit_from": "Home",
                "discharge_disposition": None,
                "discharge_location": None,
            }
        )
        if stop:
            out.append(
                {
                    **base,
                    "event_date": stop,
                    "status": "Discharged",
                    "clean_status": "DISCHARGE",
                    "admit_from": None,
                    "discharge_disposition": "Discharged to home",
                    "discharge_location": "Home",
                }
            )
    ctx.add_rows("dw.federated_adt_v", out)
