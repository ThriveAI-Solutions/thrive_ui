# scripts/sample_db/transformers/immunizations.py
"""Synthea immunizations.csv → dw.federated_vaccination_v."""

from __future__ import annotations

import pandas as pd

from scripts.sample_db.transformers.base import TransformContext

_MANUFACTURERS = ["Merck", "Sanofi", "Pfizer", "GSK", "Moderna"]


def transform_immunizations(
    imm: pd.DataFrame,
    source_map: dict[str, str],
    ctx: TransformContext,
) -> None:
    rng = ctx.rng("immunizations")
    out = []
    for i, (_, row) in enumerate(imm.iterrows()):
        source_id = source_map.get(row["PATIENT"])
        if source_id is None:
            continue
        when = pd.to_datetime(row["DATE"], utc=True).to_pydatetime().replace(tzinfo=None)
        out.append(
            {
                "source_id": source_id,
                "cvx": str(row["CODE"]),
                "ndc": f"99999-{1000 + i:04d}-01",
                "vaccine": row["DESCRIPTION"],
                "manufacturer": rng.choice(_MANUFACTURERS),
                "lot": f"LOT-{rng.randint(1000, 9999)}",
                "datetime": when,
                "status_datetime": when,
                "location_name": "Buffalo Medical Group",
                "mvx_code": "MSD",
            }
        )
    ctx.add_rows("dw.federated_vaccination_v", out)
