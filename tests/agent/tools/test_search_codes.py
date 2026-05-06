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
        CodeSearchInput(vocabulary="snomed", query="anything")  # not in literal


def test_search_codes_cvx_mmr_returns_03():
    ctx = MagicMock()
    ctx.deps = MagicMock()
    result = search_codes(ctx, CodeSearchInput(vocabulary="cvx", query="mmr"))
    assert any(m.code == "03" for m in result)
