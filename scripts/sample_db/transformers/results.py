"""Synthea observations.csv (CATEGORY=laboratory) → dw.federated_results_v.

Spec §6.3 #4: LOINC coverage ~50%, empty code_type ~22%.
"""

from __future__ import annotations

import pandas as pd

from scripts.sample_db.noise import pick_code_type
from scripts.sample_db.transformers.base import TransformContext, naive_dt, str_or_none

_EMPTY_RATE = 0.22  # share of rows with empty code_type


def transform_results(
    observations: pd.DataFrame,
    source_map: dict[str, str],
    ctx: TransformContext,
) -> None:
    rng = ctx.rng("results")
    labs = observations[observations["CATEGORY"].str.lower() == "laboratory"]
    out = []
    for _, o in labs.iterrows():
        source_id = source_map.get(o["PATIENT"])
        if source_id is None:
            continue
        when = naive_dt(o["DATE"])
        value = str_or_none(o["VALUE"])
        description = o["DESCRIPTION"]
        out.append(
            {
                "source_id": source_id,
                "code": str(o["CODE"]),
                "code_type": pick_code_type("LOINC", rng, empty_rate=_EMPTY_RATE),
                "name": description,
                "mnemonic": description[:20] if pd.notna(description) else None,
                "result": value,
                "clean_result": value,
                "unit": o["UNITS"] if pd.notna(o["UNITS"]) else None,
                "datetime": when,
                "service_provider": "Synthea Lab",
            }
        )
    ctx.add_rows("dw.federated_results_v", out)
