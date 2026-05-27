"""Polyglot code-system noise injector.

Per docs/superpowers/research/2026-05-06-redshift-warehouse-findings.md,
the real warehouse encodes the same code system under multiple spellings
(e.g. RxNorm has 4+ variants plus an OID). The ETL stamps these variants
on a configurable share of rows so the agent encounters realistic dirtiness.
"""

from __future__ import annotations

import random
from typing import Iterable

_VARIANTS: dict[str, list[str]] = {
    "RxNorm": ["RxNorm", "RXNORM", "RXNorm", "NLM RxNorm", "2.16.840.1.113883.6.88"],
    "SNOMED": ["SNOMED CT", "SNOMED-CT", "SNOMED", "SNOMEDCT", "2.16.840.1.113883.6.96"],
    "ICD-10": ["ICD-10", "ICD10", "ICD-10-CM", "ICD10CM"],
    "ICD-9": ["ICD-9", "ICD9", "ICD-9-CM"],
    "LOINC": ["LOINC", "LOINC-LN"],
    "CPT": ["CPT", "CPT4", "HCPCS"],
    "CVX": ["CVX"],
    "NDC": ["NDC"],
}


def code_type_variants_for(canonical: str) -> list[str]:
    """Return the spelling variants for a canonical code system, or [canonical]."""
    return _VARIANTS.get(canonical, [canonical])


def pick_code_type(canonical: str, rng: random.Random, *, empty_rate: float) -> str:
    """Pick one variant for the canonical code system, or '' with probability `empty_rate`."""
    if empty_rate > 0 and rng.random() < empty_rate:
        return ""
    return rng.choice(code_type_variants_for(canonical))


def inject_polyglot_code_types(
    rows: Iterable[dict],
    column: str,
    canonical: str,
    rng: random.Random,
    *,
    empty_rate: float,
) -> None:
    """In-place: replace each `row[column]` with a variant or empty string."""
    for row in rows:
        row[column] = pick_code_type(canonical, rng, empty_rate=empty_rate)
