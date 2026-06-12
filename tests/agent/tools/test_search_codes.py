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
        CodeSearchInput(vocabulary="icd9", query="anything")  # not in literal


def test_search_codes_cvx_mmr_returns_03():
    ctx = MagicMock()
    ctx.deps = MagicMock()
    result = search_codes(ctx, CodeSearchInput(vocabulary="cvx", query="mmr"))
    assert any(m.code == "03" for m in result)


def test_search_codes_snomed_penicillin_allergy_returns_code_family():
    """Epic #203 acceptance: 'penicillin allergy' through search_codes
    must surface the curated drug-allergy SNOMED subset."""
    ctx = MagicMock()
    ctx.deps = MagicMock()
    result = search_codes(ctx, CodeSearchInput(vocabulary="snomed", query="penicillin allergy"))
    codes = {m.code for m in result}
    assert "91936005" in codes, f"expected 91936005 in {codes}"
    assert "294505008" in codes, f"expected 294505008 (amoxicillin cross-reactive) in {codes}"


def test_search_codes_snomed_food_allergy_intent():
    """'any food allergy' resolves to the food-allergy bucket per #203."""
    ctx = MagicMock()
    ctx.deps = MagicMock()
    result = search_codes(ctx, CodeSearchInput(vocabulary="snomed", query="any food allergy"))
    codes = {m.code for m in result}
    assert "91934008" in codes, f"expected peanut 91934008 in {codes}"
    assert "91930004" in codes, f"expected egg 91930004 in {codes}"


def test_search_codes_snomed_peanut_allergy_specific():
    """Specific lay term 'peanut allergy' resolves to a single code."""
    ctx = MagicMock()
    ctx.deps = MagicMock()
    result = search_codes(ctx, CodeSearchInput(vocabulary="snomed", query="peanut allergy"))
    assert [m.code for m in result] == ["91934008"]
