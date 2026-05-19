# scripts/sample_db/transformers/problems.py
"""Synthea conditions.csv → dw.federated_problems_v.

Per spec §6.3 #2: warehouse is ~57% ICD-10, ~25% SNOMED, ~5% ICD-9. We
split mapped Synthea conditions accordingly.
"""

from __future__ import annotations

import pandas as pd

from scripts.sample_db.crosswalks.loader import snomed_to_icd10
from scripts.sample_db.noise import pick_code_type
from scripts.sample_db.transformers.base import TransformContext

_TARGET_ICD10 = 0.70  # of mappable rows
_TARGET_SNOMED = 0.25
# remainder → ICD-9 (best-effort, kept as SNOMED if no ICD-9 mapping)


def transform_problems(
    conditions: pd.DataFrame,
    source_map: dict[str, str],
    ctx: TransformContext,
) -> None:
    rng = ctx.rng("problems")
    cw = snomed_to_icd10()
    out = []
    for _, c in conditions.iterrows():
        source_id = source_map.get(c["PATIENT"])
        if source_id is None:
            continue
        snomed_code = str(c["CODE"])
        icd10 = cw.get(snomed_code)
        when = pd.to_datetime(c["START"]).to_pydatetime()

        # Decide which code system this row gets.
        roll = rng.random()
        if icd10 and roll < _TARGET_ICD10:
            code, canonical = icd10, "ICD-10"
        elif roll < _TARGET_ICD10 + _TARGET_SNOMED:
            code, canonical = snomed_code, "SNOMED"
        elif icd10:
            # Drop final digit to fake an ICD-9-ish look. Not technically
            # correct, but represents the legacy bucket the agent has to
            # tolerate.
            code, canonical = icd10[:-1] if "." in icd10 else icd10, "ICD-9"
        else:
            code, canonical = snomed_code, "SNOMED"

        out.append(
            {
                "source_id": source_id,
                "code": code,
                "code_type": pick_code_type(canonical, rng, empty_rate=0.05),
                "diagnosis": c["DESCRIPTION"],
                "diagnosis_datetime": when,
                "status_datetime": when,
                "chronic_ind": "Y" if rng.random() < 0.4 else "N",
                "service_provider_npi": None,
            }
        )
    ctx.add_rows("dw.federated_problems_v", out)
