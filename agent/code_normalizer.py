"""Map canonical vocabulary tokens to the warehouse code_type spelling variants.

Per spec §7.12 reliability notes, the warehouse `code_type` column is
inconsistently populated. SQL templates that filter on a vocabulary should
expand a canonical token into the IN-list of observed spellings via
`variants_for(canonical)`.

The reverse direction (`normalize_token`) is used when reading rows back
from the warehouse so callers can group/badge by canonical vocab.
"""

from __future__ import annotations
from typing import Optional


_VARIANTS: dict[str, tuple[str, ...]] = {
    "loinc": ("LOINC", "loinc", "Loinc", "LN", "LOINC-2.69", "LOINC-2.70"),
    "icd10": ("ICD-10", "ICD10", "icd10", "ICD-10-CM", "ICD10-CM", "ICD-10-PCS"),
    "icd9": ("ICD-9", "ICD9", "ICD-9-CM"),
    "snomed": ("SNOMED", "snomed", "SNOMED-CT", "SCT"),
    "cvx": ("CVX", "cvx"),
    "rxnorm": ("RXNORM", "rxnorm", "RxNorm"),
    "cpt": ("CPT", "CPT4", "cpt", "CPT-4"),
    "ndc": ("NDC", "ndc"),
}


def variants_for(canonical: str) -> tuple[str, ...]:
    """Return all warehouse code_type spellings mapped to the canonical token."""
    key = canonical.lower()
    if key not in _VARIANTS:
        raise ValueError(f"unknown vocabulary: {canonical!r}")
    return _VARIANTS[key]


def normalize_token(observed: Optional[str]) -> Optional[str]:
    """Reverse: take a row's `code_type` value, return the canonical token or None."""
    if observed is None:
        return None
    for canonical, variants in _VARIANTS.items():
        if observed in variants:
            return canonical
    return None
