# tests/agent/codes/test_allergies.py
"""Tests for the curated allergy SNOMED code allow-list (Epic #203).

The module is the shared source of truth for allergy SNOMED codes. These
tests pin the categories the epic ships with and the contract of the
helpers that other tools consume.

Prior to Task 3 (2026-07-03 vocab port) this file also carried a drift
guard asserting every curated code exists in the (now-deleted)
agent/codes/data/snomed.json. That static fixture file no longer ships —
vocabulary data lives in the vocab_* DB tables, populated from a chiron
export (see scripts/import_vocab_dump.py) rather than a file committed to
this repo — so there's no static artifact left to diff the allow-list
against here. The equivalent check now belongs with the vocab import
tooling (sanity floors in scripts/import_vocab_dump.py), not this module.
"""

from __future__ import annotations

import pytest

from agent.codes.allergies import (
    ALLERGY_CATEGORIES,
    ALLERGY_SNOMED_BY_CATEGORY,
    all_allergy_codes,
    codes_for_category,
)


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
    assert len(flat) == len(set(flat)), f"duplicate codes across categories: {[c for c in flat if flat.count(c) > 1]}"


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
