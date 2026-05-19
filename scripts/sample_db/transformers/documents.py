"""Encounters → dw.federated_documents_v (index only, no bodies).

Real warehouse holds doc metadata; bodies live in HEALTHeLINK. We mirror
the index structure with realistic name/mnemonic distributions from the
warehouse findings doc.
"""

from __future__ import annotations

import pandas as pd

from scripts.sample_db.transformers.base import TransformContext, naive_dt

# (name, mnemonic, weight) from 2026-05-06-redshift-warehouse-findings.md §4.
_DOC_TYPES = [
    ("Progress Note", "PROGNOTE", 30),
    ("Visit Note", "VISITNOTE", 20),
    ("ED MD Note", "EDMDNOTE", 18),
    ("H&P", "HPNOTE", 8),
    ("Consult", "CONSULT", 7),
    ("Discharge Summary", "DCSUMM", 5),
    ("Operative Note", "OPNOTE", 4),
    ("Anesthesia Note", "ANESTNOTE", 4),
    ("Radiology Report", "XRREPORT", 4),
]


def _weighted_choice(rng):
    total = sum(w for _, _, w in _DOC_TYPES)
    r = rng.uniform(0, total)
    acc = 0
    for name, mnem, w in _DOC_TYPES:
        acc += w
        if r <= acc:
            return name, mnem
    return _DOC_TYPES[0][:2]


def transform_documents(
    encounters: pd.DataFrame,
    source_map: dict[str, str],
    ctx: TransformContext,
) -> None:
    rng = ctx.rng("documents")
    out = []
    for _, e in encounters.iterrows():
        source_id = source_map.get(e["PATIENT"])
        if source_id is None:
            continue
        when = naive_dt(e["START"])
        name, mnem = _weighted_choice(rng)
        out.append(
            {
                "source_id": source_id,
                "datetime": when,
                "name": name,
                "mnemonic": mnem,
                "status": "final",
                "encounter_id": e["Id"],
                "place_of_service": "11",
                "location_name": e["ORGANIZATION"] if pd.notna(e["ORGANIZATION"]) else "Unknown",
            }
        )
    ctx.add_rows("dw.federated_documents_v", out)
