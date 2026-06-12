# tests/agent/codes/test_allergies.py
"""Tests for the curated allergy SNOMED code allow-list (Epic #203).

The module is the shared source of truth for allergy SNOMED codes. These
tests pin the categories the epic ships with, the contract of the helpers
that other tools will consume, and the invariant that every curated code
exists in agent/codes/data/snomed.json (drift guard).
"""

from __future__ import annotations

import pytest

from agent.codes.allergies import (
    ALLERGY_CATEGORIES,
    ALLERGY_SNOMED_BY_CATEGORY,
    all_allergy_codes,
    codes_for_category,
)
from agent.codes.loader import VocabLoader


EXPECTED_CATEGORIES = ("drug", "food", "environmental", "contact", "anaphylaxis")


def test_categories_match_epic_203():
    assert tuple(ALLERGY_SNOMED_BY_CATEGORY.keys()) == EXPECTED_CATEGORIES
    assert ALLERGY_CATEGORIES == EXPECTED_CATEGORIES


def test_every_category_is_non_empty():
    for category, codes in ALLERGY_SNOMED_BY_CATEGORY.items():
        assert codes, f"category {category!r} has no codes — empty buckets are not allowed"


def test_codes_for_category_returns_independent_copy():
    """codes_for_category must hand callers a mutable copy so they can
    splice/filter without poisoning the module-level table."""
    first = codes_for_category("drug")
    first.append("FAKE")
    second = codes_for_category("drug")
    assert "FAKE" not in second


def test_codes_for_category_unknown_raises():
    with pytest.raises(ValueError, match="unknown allergy category"):
        codes_for_category("not-a-category")


def test_all_allergy_codes_is_union_in_category_order():
    flat = all_allergy_codes()
    expected = [c for codes in ALLERGY_SNOMED_BY_CATEGORY.values() for c in codes]
    assert flat == expected


def test_all_allergy_codes_have_no_cross_category_duplicates():
    """If a code lands in two categories that's a curation mistake — pick
    the primary category. Reviewers will rely on this invariant."""
    flat = all_allergy_codes()
    assert len(flat) == len(set(flat)), (
        f"duplicate codes across categories: "
        f"{[c for c in flat if flat.count(c) > 1]}"
    )


def test_drift_guard_every_curated_code_exists_in_snomed_data():
    """Every code in ALLERGY_SNOMED_BY_CATEGORY must exist in the
    snomed.json data file — otherwise search_codes will silently drop
    it via its phantom-code filter and downstream tools will end up with
    a quietly empty result."""
    data_codes = {row["code"] for row in VocabLoader().entries("snomed")}
    missing = [c for c in all_allergy_codes() if c not in data_codes]
    assert not missing, f"curated codes missing from snomed.json: {missing}"


def test_drug_category_includes_penicillin_and_amoxicillin():
    """Spot-check: the drug bucket must include both penicillin (91936005)
    and amoxicillin (294505008) — the agent uses this pairing for the
    cross-reactive answer to 'penicillin allergy'."""
    drug = codes_for_category("drug")
    assert "91936005" in drug
    assert "294505008" in drug


def test_anaphylaxis_category_includes_systemic_codes():
    anaph = codes_for_category("anaphylaxis")
    assert "39579001" in anaph  # Anaphylaxis
    assert "241929008" in anaph  # Acute allergic reaction
