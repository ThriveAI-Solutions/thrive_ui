# tests/agent/tools/test_search_codes.py
from unittest.mock import MagicMock
import pytest

from agent.tools.search_codes import (
    search_codes,
    CodeSearchInput,
    CodeMatch,
)


def test_search_codes_loinc_a1c():
    ctx = MagicMock()
    ctx.deps = MagicMock()
    result = search_codes(ctx, CodeSearchInput(vocabulary="loinc", query="hemoglobin a1c", limit=5))
    assert isinstance(result, list)
    assert len(result) >= 1
    assert all(isinstance(m, CodeMatch) for m in result)
    assert any("a1c" in m.display_name.lower() for m in result)


def test_search_codes_limit_clamped_to_50():
    ctx = MagicMock()
    ctx.deps = MagicMock()
    # Pydantic validates the input model; passing limit > 50 must raise.
    with pytest.raises(Exception):
        CodeSearchInput(vocabulary="loinc", query="", limit=51)


def test_search_codes_unknown_vocabulary_raises_validation():
    ctx = MagicMock()
    ctx.deps = MagicMock()
    with pytest.raises(Exception):
        CodeSearchInput(vocabulary="made_up_vocab", query="anything")  # not in literal


def test_search_codes_cvx_mmr_returns_03():
    ctx = MagicMock()
    ctx.deps = MagicMock()
    result = search_codes(ctx, CodeSearchInput(vocabulary="cvx", query="mmr"))
    assert any(m.code == "03" for m in result)


def test_search_codes_snomed_penicillin_allergy_returns_code():
    """Per epic #201: SNOMED vocabulary covers allergy codes. The substring
    fallback should find 'penicillin' in the allergen display name."""
    ctx = MagicMock()
    ctx.deps = MagicMock()
    result = search_codes(ctx, CodeSearchInput(vocabulary="snomed", query="penicillin"))
    assert any(m.code == "91936005" for m in result)


def test_search_codes_snomed_drug_allergy_synonym_returns_full_class():
    """Synonyms.json routes 'drug allergy' intent → SNOMED drug-allergy code set."""
    ctx = MagicMock()
    ctx.deps = MagicMock()
    result = search_codes(ctx, CodeSearchInput(vocabulary="snomed", query="drug allergy"))
    codes = {m.code for m in result}
    # All drug-allergy SNOMED codes from agent.codes.allergies should be present.
    assert "91936005" in codes  # penicillin
    assert "91937001" in codes  # sulfonamide
