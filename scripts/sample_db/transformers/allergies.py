"""Synthea allergies.csv → dw.federated_allergies_v.

Per epic #201: structured allergy records, surfaced as a first-class
domain by the agent. Synthea's CATEGORY column ('medication' / 'food' /
'environment' / 'biologic') is normalized to the canonical warehouse
`type` values observed in the dev-warehouse eval set ('Drug allergy' /
'Food allergy' / 'Environmental allergy' / 'Adverse Reaction').
"""

from __future__ import annotations

import pandas as pd

from scripts.sample_db.noise import pick_code_type
from scripts.sample_db.transformers.base import TransformContext


_CATEGORY_MAP = {
    "medication": "Drug allergy",
    "drug": "Drug allergy",
    "food": "Food allergy",
    "environment": "Environmental allergy",
    "environmental": "Environmental allergy",
    "biologic": "Adverse Reaction",
}


def _normalize_allergen(description: str) -> str:
    """Strip Synthea's "Allergy to " prefix so the `allergy` column carries
    the bare allergen name ("Penicillin"), matching the eval-set wording."""
    if not isinstance(description, str):
        return description
    if description.lower().startswith("allergy to "):
        # Title-case the first letter; preserve hyphenation/proper nouns.
        tail = description[len("Allergy to ") :].strip()
        return tail[:1].upper() + tail[1:] if tail else description
    return description


def _normalize_severity(value: object) -> object:
    if not isinstance(value, str):
        return None
    return value.strip().title() if value.strip() else None


def transform_allergies(
    allergies: pd.DataFrame,
    source_map: dict[str, str],
    ctx: TransformContext,
) -> None:
    rng = ctx.rng("allergies")
    out: list[dict] = []
    for _, a in allergies.iterrows():
        source_id = source_map.get(a["PATIENT"])
        if source_id is None:
            continue
        start = pd.to_datetime(a["START"]).to_pydatetime() if pd.notna(a.get("START")) else None
        stop = pd.to_datetime(a["STOP"]).to_pydatetime() if pd.notna(a.get("STOP")) else None
        category_raw = (a.get("CATEGORY") or "").strip().lower() if isinstance(a.get("CATEGORY"), str) else ""
        out.append(
            {
                "source_id": source_id,
                "code": str(a["CODE"]) if pd.notna(a.get("CODE")) else None,
                "code_type": pick_code_type("SNOMED", rng, empty_rate=0.05),
                "allergy": _normalize_allergen(a.get("DESCRIPTION") or ""),
                "type": _CATEGORY_MAP.get(category_raw, "Adverse Reaction"),
                "severity": _normalize_severity(a.get("SEVERITY1")),
                "status": "Resolved" if stop else "Active",
                "onset_date": start,
                "status_datetime": stop or start,
                "reaction": a.get("DESCRIPTION1") if isinstance(a.get("DESCRIPTION1"), str) else None,
                "comments": None,
            }
        )
    ctx.add_rows("dw.federated_allergies_v", out)
