# scripts/sample_db/transformers/encounters.py
"""Synthea encounters.csv → dw.federated_encounters_v."""

from __future__ import annotations

import pandas as pd

from scripts.sample_db.transformers.base import TransformContext, naive_dt

_CLASS_MAP = {
    "ambulatory": "office_visit",
    "wellness": "office_visit",
    "outpatient": "office_visit",
    "urgentcare": "office_visit",
    "inpatient": "inpatient",
    "emergency": "emergency",
    "home": "home_visit",
    "snf": "snf",
    "hospice": "hospice",
}

_POS_MAP = {
    "office_visit": "11",
    "emergency": "23",
    "inpatient": "21",
    "home_visit": "12",
    "snf": "31",
    "hospice": "34",
}


def transform_encounters(
    encounters: pd.DataFrame,
    source_map: dict[str, str],
    ctx: TransformContext,
) -> None:
    out = []
    for _, e in encounters.iterrows():
        synthea_patient = e["PATIENT"]
        source_id = source_map.get(synthea_patient)
        if source_id is None:
            continue
        enc_class = str(e.get("ENCOUNTERCLASS", "")).lower()
        wh_type = _CLASS_MAP.get(enc_class, "office_visit")
        when = naive_dt(e["START"])
        out.append(
            {
                "source_id": source_id,
                "encounter_id": e["Id"],
                "type": wh_type,
                "status": "completed",
                "status_datetime": when,
                "datetime": when,
                "location": str(e.get("ORGANIZATION", "")) or None,
                "rendering_provider": str(e.get("PROVIDER", "")) or None,
                "facility_name": str(e.get("ORGANIZATION", "")) or None,
                "place_of_service": _POS_MAP.get(wh_type, "11"),
            }
        )
    ctx.add_rows("dw.federated_encounters_v", out)
