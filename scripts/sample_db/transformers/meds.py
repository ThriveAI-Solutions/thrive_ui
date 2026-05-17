# scripts/sample_db/transformers/meds.py
"""Synthea medications.csv → dw.federated_meds_v.

Real warehouse has both ndc_code and rxnorm_code at 100%. Synthea emits
only RxNorm; crosswalk to NDC, fall back to a documented placeholder so
the NDC column stays 100% populated.
"""

from __future__ import annotations

import pandas as pd

from scripts.sample_db.crosswalks.loader import rxnorm_to_ndc
from scripts.sample_db.transformers.base import TransformContext

_PLACEHOLDER_NDC = "99999-9999-99"


def transform_meds(
    meds: pd.DataFrame,
    source_map: dict[str, str],
    ctx: TransformContext,
) -> None:
    rng = ctx.rng("meds")
    cw = rxnorm_to_ndc()
    out = []
    for _, m in meds.iterrows():
        source_id = source_map.get(m["PATIENT"])
        if source_id is None:
            continue
        rxnorm = str(m["CODE"])
        ndc = cw.get(rxnorm, _PLACEHOLDER_NDC)
        when = pd.to_datetime(m["START"], utc=True).to_pydatetime().replace(tzinfo=None)
        desc = str(m["DESCRIPTION"]) if pd.notna(m["DESCRIPTION"]) else None
        strength, unit = _parse_strength(desc) if desc else (None, None)
        out.append(
            {
                "source_id": source_id,
                "ndc_code": ndc,
                "rxnorm_code": rxnorm,
                "med_name": desc,
                "date_prescribed": when,
                "prescribing_provider_npi": None,
                "med_strength": strength,
                "med_strength_unit": unit,
                "med_form": "Tab",
                "med_sig": "Take as directed",
                "drug_supply_days": int(rng.choice([30, 60, 90])),
                "number_of_refills": int(rng.choice([0, 1, 2, 3])),
            }
        )
    ctx.add_rows("dw.federated_meds_v", out)


def _parse_strength(desc: str) -> tuple[str | None, str | None]:
    """Extract '500' + 'MG' from 'Metformin 500 MG'."""
    parts = desc.split()
    for i, p in enumerate(parts):
        if p.replace(".", "", 1).isdigit() and i + 1 < len(parts):
            unit = parts[i + 1].upper()
            if unit in {"MG", "MCG", "G", "ML", "UNIT", "IU"}:
                return p, unit
    return None, None
