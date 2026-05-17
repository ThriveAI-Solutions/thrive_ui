"""Synthea procedures.csv → dw.federated_orders_v.

Spec §6.3 #4: ~46% of orders have empty code_type in real warehouse.
We mark most rows as CPT to match the production distribution where
orders are clinical-coded.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from scripts.sample_db.noise import pick_code_type
from scripts.sample_db.transformers.base import TransformContext

_EMPTY_RATE = 0.46


def transform_orders(
    procedures: pd.DataFrame,
    source_map: dict[str, str],
    ctx: TransformContext,
) -> None:
    rng = ctx.rng("orders")
    out = []
    for _, p in procedures.iterrows():
        source_id = source_map.get(p["PATIENT"])
        if source_id is None:
            continue
        start = pd.to_datetime(p["START"], utc=True).to_pydatetime().replace(tzinfo=None)
        created = start - dt.timedelta(hours=int(rng.choice([0, 24, 72, 168])))
        out.append(
            {
                "source_id": source_id,
                "code": str(p["CODE"]),
                "code_type": pick_code_type("CPT", rng, empty_rate=_EMPTY_RATE),
                "name": p["DESCRIPTION"],
                "date_of_procedure": start,
                "order_created_date": created,
                "place_of_service": rng.choice(["11", "22"]),
                "status": "completed",
            }
        )
    ctx.add_rows("dw.federated_orders_v", out)
