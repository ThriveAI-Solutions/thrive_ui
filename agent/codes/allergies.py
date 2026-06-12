"""Allergy SNOMED code allow-list and drug-allergy conflict crosswalk.

Per epic #201:
- ALLERGY_SNOMED_BY_CATEGORY drives the `federated_problems_v` fallback path
  (when `federated_allergies_v` is unavailable, allergies are identified by
  SNOMED code allow-list + text LIKE on the diagnosis column).
- DRUG_ALLERGY_CONFLICTS feeds the soft conflict signal: when a patient has
  a recorded drug allergy AND an active medication in the same class, the
  allergies tool result emits a notes_to_agent advisory. NOT a CDS verdict.

The crosswalks are deliberately small and hand-curated. Expand by class as
real coverage gaps appear.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List


# SNOMED allergy codes grouped by clinical category. Used by the
# federated_problems_v fallback path and by the search_codes intent shortcut.
ALLERGY_SNOMED_BY_CATEGORY: dict[str, list[str]] = {
    "drug": [
        "91936005",  # Allergy to penicillin
        "91937001",  # Allergy to sulfonamide
        "294505008",  # Allergy to amoxicillin
        "293584003",  # Allergy to aspirin
        "293586001",  # Allergy to NSAID
    ],
    "food": [
        "91934008",  # Allergy to peanut
        "91935009",  # Allergy to nut
        "300913006",  # Allergy to seafood
        "417532002",  # Allergy to fish
        "91938006",  # Allergy to dairy
        "91930004",  # Allergy to egg
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
    "adverse_reaction": [
        "241929008",  # Acute allergic reaction
        "39579001",  # Anaphylaxis
    ],
}


# Free-text allergen substrings used by the federated_problems_v fallback
# path. Hits trigger an allergy classification when no SNOMED code matches.
ALLERGY_TEXT_PATTERNS: list[str] = [
    "allerg",
    "hypersensitiv",
    "intoleran",
    "anaphyla",
]


@dataclass(frozen=True)
class DrugAllergyClass:
    """A class of drugs that conflict with a specific allergen."""

    allergen_label: str
    rxnorm_codes: List[str]
    text_match: str  # case-insensitive substring on the allergy text


# Allergen SNOMED code → conflicting RxNorm drug-class codes. Curated, NOT
# pharmacologically exhaustive. Add entries as concrete clinical scenarios
# surface during eval rather than guessing at completeness.
DRUG_ALLERGY_CONFLICTS: dict[str, DrugAllergyClass] = {
    "91936005": DrugAllergyClass(
        allergen_label="Penicillin",
        rxnorm_codes=["723", "7980", "1665", "2191"],  # amox, pen G, ampicillin, augmentin
        text_match="penicillin",
    ),
    "294505008": DrugAllergyClass(
        allergen_label="Amoxicillin",
        rxnorm_codes=["723", "2191"],
        text_match="amoxicillin",
    ),
    "91937001": DrugAllergyClass(
        allergen_label="Sulfonamide",
        rxnorm_codes=["10180", "10829", "1840"],  # sulfamethoxazole, trimethoprim, sulfasalazine
        text_match="sulfa",
    ),
    "293584003": DrugAllergyClass(
        allergen_label="Aspirin",
        rxnorm_codes=["1191"],  # aspirin
        text_match="aspirin",
    ),
}


@dataclass(frozen=True)
class DrugAllergyConflict:
    allergen_label: str
    conflicting_med_name: str
    conflicting_rxnorm: str


def find_drug_allergy_conflicts(
    *,
    allergies: list[dict],
    medications: list[dict],
) -> List[DrugAllergyConflict]:
    """Pair each drug allergy against the active med list. Returns soft
    conflicts only — does not encode reactivity-graph nuance.

    Match strategy:
    1. Exact SNOMED code match (allergen.code → DrugAllergyClass)
    2. Text fallback (allergen text contains the class.text_match substring)
    """
    if not allergies or not medications:
        return []

    # Build the candidate classes for this patient's allergies.
    candidate_classes: list[tuple[str, DrugAllergyClass]] = []
    for allergy in allergies:
        code = (allergy.get("code") or "").strip()
        text = (allergy.get("allergy") or "").lower()

        # 1. Code-driven match.
        if code in DRUG_ALLERGY_CONFLICTS:
            candidate_classes.append(
                (allergy.get("allergy") or DRUG_ALLERGY_CONFLICTS[code].allergen_label, DRUG_ALLERGY_CONFLICTS[code])
            )
            continue
        # 2. Text fallback — first class whose text_match appears in the allergy text.
        for klass in DRUG_ALLERGY_CONFLICTS.values():
            if klass.text_match in text:
                candidate_classes.append((allergy.get("allergy") or klass.allergen_label, klass))
                break

    if not candidate_classes:
        return []

    conflicts: list[DrugAllergyConflict] = []
    for allergen_label, klass in candidate_classes:
        for med in medications:
            rxnorm = (med.get("rxnorm_code") or "").strip()
            if rxnorm and rxnorm in klass.rxnorm_codes:
                conflicts.append(
                    DrugAllergyConflict(
                        allergen_label=allergen_label,
                        conflicting_med_name=med.get("med_name") or "(unnamed)",
                        conflicting_rxnorm=rxnorm,
                    )
                )
    return conflicts
