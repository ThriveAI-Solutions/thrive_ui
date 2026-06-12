"""Curated allergy SNOMED code allow-list, grouped by clinical category.

This module is the shared source of truth for allergy SNOMED codes used
across the agent. It backs two consumers:

1. `search_codes` — the allergy-intent shortcut (e.g., "penicillin allergy",
   "any food allergy") routes through `synonyms.json` for the `snomed`
   vocabulary to the curated subsets defined here.
2. Future allergy-domain tooling on `get_patient_clinical_data` and the
   `federated_problems_v` fallback path (Epic #201) — when the dedicated
   `federated_allergies_v` view is unavailable, the agent identifies
   allergies in the problem list via this allow-list.

The categories follow Epic #203:

- drug — antibiotic / analgesic class allergies
- food — IgE-mediated food allergens
- environmental — pollens, allergic rhinitis variants
- contact — latex and other contact-route allergens
- anaphylaxis — life-threatening / acute systemic reactions

Hand-curated and intentionally small. Expand by category as concrete
coverage gaps surface during agent evaluation rather than guessing at
completeness.
"""

from __future__ import annotations

from typing import Final


ALLERGY_SNOMED_BY_CATEGORY: Final[dict[str, list[str]]] = {
    "drug": [
        "91936005",   # Allergy to penicillin
        "294505008",  # Allergy to amoxicillin
        "91937001",   # Allergy to sulfonamide
        "293584003",  # Allergy to aspirin
        "293586001",  # Allergy to NSAID
    ],
    "food": [
        "91934008",   # Allergy to peanut
        "91935009",   # Allergy to nut
        "300913006",  # Allergy to seafood
        "417532002",  # Allergy to fish
        "91938006",   # Allergy to dairy
        "91930004",   # Allergy to eggs
        "300915002",  # Allergy to wheat
    ],
    "environmental": [
        "232347008",  # Allergic rhinitis due to pollen
        "446096008",  # Perennial allergic rhinitis
        "367498001",  # Seasonal allergic rhinitis
        "232353008",  # Perennial allergic rhinitis with seasonal variation
        "418689008",  # Allergy to grass pollen
    ],
    "contact": [
        "300916003",  # Allergy to latex
    ],
    "anaphylaxis": [
        "39579001",   # Anaphylaxis
        "241929008",  # Acute allergic reaction
    ],
}


ALLERGY_CATEGORIES: Final[tuple[str, ...]] = tuple(ALLERGY_SNOMED_BY_CATEGORY.keys())


def codes_for_category(category: str) -> list[str]:
    """Return the SNOMED codes for a single allergy category.

    Raises ValueError on unknown category so callers don't silently
    drop a typo into a successful empty lookup.
    """
    if category not in ALLERGY_SNOMED_BY_CATEGORY:
        raise ValueError(
            f"unknown allergy category {category!r}; expected one of {ALLERGY_CATEGORIES}"
        )
    return list(ALLERGY_SNOMED_BY_CATEGORY[category])


def all_allergy_codes() -> list[str]:
    """Return every curated allergy SNOMED code, in category order."""
    return [code for codes in ALLERGY_SNOMED_BY_CATEGORY.values() for code in codes]
