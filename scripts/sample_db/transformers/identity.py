# scripts/sample_db/transformers/identity.py
"""Transform Synthea patients.csv + encounters.csv -> 4 identity tables.

Tables produced:
- dw.internal_patient_profile_v
- dw.internal_source_reference_v
- dw.federated_demographic_v
- dw.federated_demographic_history_v
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from scripts.sample_db.transformers.base import TransformContext

_INACTIVE_RANK_RATE = 0.10  # ~10% of patients get an extra empi_rank=99 source
_STALE_VISIT_RATE = 0.15  # ~15% have last_date_of_visit >12 months ago
_OUT_OF_WNY_RATE = 0.08  # ~8% live outside Buffalo area


def _age(birthdate: str, ref_date: dt.date) -> int:
    bd = dt.date.fromisoformat(birthdate)
    return ref_date.year - bd.year - ((ref_date.month, ref_date.day) < (bd.month, bd.day))


def transform_identity(
    patients: pd.DataFrame,
    encounters: pd.DataFrame,
    ctx: TransformContext,
) -> None:
    rng = ctx.rng("identity")
    ref_date = dt.date.today()
    stale_date = ref_date - dt.timedelta(days=400)

    # Pre-compute last encounter date per Synthea patient.
    enc = encounters.copy()
    enc["START"] = pd.to_datetime(enc["START"], utc=True).dt.date
    last_visit_by_patient = enc.groupby("PATIENT")["START"].max().to_dict()

    # Pre-compute provider/org for the latest encounter per patient.
    enc_sorted = enc.sort_values(["PATIENT", "START"])
    last_enc_by_patient = enc_sorted.groupby("PATIENT").tail(1).set_index("PATIENT")

    profiles, refs, demo, demo_hist = [], [], [], []

    for int_id, (_, p) in enumerate(patients.iterrows(), start=1):
        synthea_id = p["Id"]
        first, last = p["FIRST"], p["LAST"]
        gender = p["GENDER"]
        dob = p["BIRTHDATE"]
        age = _age(dob, ref_date)

        # Last visit / freshness.
        lv = last_visit_by_patient.get(synthea_id)
        if lv is None:
            last_visit = None
        elif rng.random() < _STALE_VISIT_RATE:
            last_visit = stale_date
        else:
            last_visit = lv

        # Practice / provider.
        if synthea_id in last_enc_by_patient.index:
            practice = str(last_enc_by_patient.loc[synthea_id, "ORGANIZATION"])
            provider = str(last_enc_by_patient.loc[synthea_id, "PROVIDER"])
        else:
            practice, provider = None, None

        # Geography -- keep most patients in Buffalo, scatter a small fraction.
        if rng.random() < _OUT_OF_WNY_RATE:
            zip_code, city, state = p["ZIP"], p["CITY"], p["STATE"]  # whatever Synthea gave
        else:
            zip_code, city, state = "14223", "Buffalo", "NY"

        profiles.append(
            {
                "patient_id": int_id,
                "first_name": first,
                "last_name": last,
                "full_name": f"{first} {last}",
                "date_of_birth": dob,
                "age": age,
                "gender": gender,
                "last_date_of_visit": last_visit,
                "practice_name": practice,
                "provider_name": provider,
                "conditions": None,  # populated later if rollup transformer wants to
                "zip_code": zip_code,
                "city": city,
                "state": state,
            }
        )

        # Source references -- one canonical, optional stale rank=99 row.
        primary_source = f"src-{synthea_id}"
        refs.append(
            {
                "patient_id": int_id,
                "source_id": primary_source,
                "empi_rank": 1,
                "source_name": practice or "Buffalo Medical Group",
                "source_type": "EHR",
            }
        )
        if rng.random() < _INACTIVE_RANK_RATE:
            refs.append(
                {
                    "patient_id": int_id,
                    "source_id": f"{primary_source}-stale",
                    "empi_rank": 99,
                    "source_name": practice or "Buffalo Medical Group",
                    "source_type": "EHR",
                }
            )

        # Federated demographic (one row per active source).
        demo.append(
            {
                "source_id": primary_source,
                "patient_id": str(int_id),
                "first_name": first,
                "last_name": last,
                "date_of_birth": dob,
                "gender": gender,
            }
        )
        # Demographic history: snapshot today.
        demo_hist.append(
            {
                "source_id": primary_source,
                "patient_id": str(int_id),
                "first_name": first,
                "last_name": last,
                "date_of_birth": dob,
                "gender": gender,
            }
        )

    ctx.add_rows("dw.internal_patient_profile_v", profiles)
    ctx.add_rows("dw.internal_source_reference_v", refs)
    ctx.add_rows("dw.federated_demographic_v", demo)
    ctx.add_rows("dw.federated_demographic_history_v", demo_hist)
