"""Load crosswalk CSVs into lookup dicts. Single-file, no DB dependency."""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

_CROSSWALK_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=None)
def snomed_to_icd10() -> dict[str, str]:
    """Return {snomed_code: icd10_code}. Empty for unmapped."""
    return _load_first_two_cols("snomed_to_icd10.csv")


@lru_cache(maxsize=None)
def rxnorm_to_ndc() -> dict[str, str]:
    """Return {rxnorm_code: ndc_code}. Empty for unmapped."""
    return _load_first_two_cols("rxnorm_to_ndc.csv")


def _load_first_two_cols(filename: str) -> dict[str, str]:
    path = _CROSSWALK_DIR / filename
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader)  # skip header
        return {row[0].strip(): row[1].strip() for row in reader if row}
