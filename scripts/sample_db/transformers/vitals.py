"""Synthea observations.csv (CATEGORY=vital-signs) → dw.federated_vitals_v.

Spec §6.3: vitals are the cleanest dataset, LOINC ~91% with ~7% empty.
"""

from __future__ import annotations

import pandas as pd

from scripts.sample_db.noise import pick_code_type
from scripts.sample_db.transformers.base import TransformContext, naive_dt, str_or_none

_EMPTY_RATE = 0.07


def transform_vitals(
    observations: pd.DataFrame,
    source_map: dict[str, str],
    ctx: TransformContext,
) -> None:
    rng = ctx.rng("vitals")
    vitals = observations[observations["CATEGORY"].str.lower() == "vital-signs"]
    out = []
    for _, o in vitals.iterrows():
        source_id = source_map.get(o["PATIENT"])
        if source_id is None:
            continue
        when = naive_dt(o["DATE"])
        value = str_or_none(o["VALUE"])
        out.append(
            {
                "source_id": source_id,
                "code": str(o["CODE"]),
                "code_type": pick_code_type("LOINC", rng, empty_rate=_EMPTY_RATE),
                "name": o["DESCRIPTION"],
                "result": value,
                "clean_result": value,
                "unit": o["UNITS"] if pd.notna(o["UNITS"]) else None,
                "datetime": when,
            }
        )
    ctx.add_rows("dw.federated_vitals_v", out)
