# tests/agent/codes/test_loader.py
import pytest
from agent.codes.loader import VocabLoader, search


def test_loader_loads_all_five_vocabularies():
    loader = VocabLoader()
    for vocab in ("loinc", "cvx", "rxnorm", "icd10", "cpt"):
        rows = loader.entries(vocab)
        assert len(rows) >= 20, f"{vocab} has only {len(rows)} entries"


def test_loader_unknown_vocabulary_raises():
    loader = VocabLoader()
    with pytest.raises(ValueError, match="unknown vocabulary"):
        loader.entries("snomed-fake")


def test_search_loinc_by_display_name():
    results = search(vocabulary="loinc", query="hemoglobin a1c", limit=5)
    assert len(results) >= 1
    assert any("a1c" in r.display_name.lower() for r in results)


def test_search_cvx_by_common_name_mmr():
    results = search(vocabulary="cvx", query="mmr", limit=5)
    assert len(results) >= 1
    assert any(r.code == "03" for r in results)


def test_search_icd10_by_diabetes():
    results = search(vocabulary="icd10", query="diabetes", limit=10)
    assert any(r.code.startswith("E11") for r in results)


def test_search_rxnorm_by_azithromycin():
    results = search(vocabulary="rxnorm", query="azithromycin", limit=5)
    assert len(results) >= 1
    assert any("azith" in r.display_name.lower() for r in results)


def test_search_cpt_by_colonoscopy():
    results = search(vocabulary="cpt", query="colonoscopy", limit=5)
    assert any(r.code == "45378" for r in results)


def test_search_limit_respected():
    results = search(vocabulary="loinc", query="", limit=3)
    assert len(results) <= 3


def test_search_exact_code_lookup():
    results = search(vocabulary="cvx", query="03", limit=5)
    assert any(r.code == "03" for r in results)
