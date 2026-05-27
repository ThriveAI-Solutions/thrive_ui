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


# ---------------------------------------------------------------------------
# Synonym + code-family layer (agent/codes/data/synonyms.json)
#
# These tests exercise the abbreviation / colloquial → canonical → code list
# resolution that lets doctors typing "HTN" and patients typing "high blood
# pressure" hit the same hypertension code family. Substring fallback for
# technical terms not in the synonym map is preserved.
# ---------------------------------------------------------------------------


def test_synonym_layer_colloquial_resolves_to_canonical_family():
    """'high blood pressure' is not in any canonical display name but
    should resolve via the synonym map to the I10/I11/I12/I13 family."""
    results = search(vocabulary="icd10", query="high blood pressure", limit=20)
    codes = {r.code for r in results}
    assert "I10" in codes, f"expected I10 in {codes}"
    # Family expansion should bring at least one I11/I12/I13 family code.
    assert any(c.startswith(("I11", "I12", "I13", "I16")) for c in codes), (
        f"expected hypertension family beyond I10; got {codes}"
    )


def test_synonym_layer_abbreviation_htn():
    """Doctor types 'HTN' — must resolve to the hypertension family
    even though no display name contains the literal string 'HTN'."""
    results = search(vocabulary="icd10", query="HTN", limit=20)
    codes = {r.code for r in results}
    assert "I10" in codes, f"HTN must resolve to I10; got {codes}"


def test_synonym_layer_abbreviation_dm_t2dm():
    """'DM' and 'T2DM' are common doctor abbreviations for diabetes."""
    for needle in ("DM", "T2DM", "t2dm", "sugar"):
        results = search(vocabulary="icd10", query=needle, limit=20)
        codes = {r.code for r in results}
        assert "E11.9" in codes, f"{needle!r} must resolve to E11.9; got {codes}"


def test_synonym_layer_chf_resolves_to_heart_failure_family():
    results = search(vocabulary="icd10", query="chf", limit=20)
    codes = {r.code for r in results}
    assert "I50.9" in codes, f"CHF must resolve to heart-failure family; got {codes}"


def test_synonym_layer_lay_term_heart_attack():
    results = search(vocabulary="icd10", query="heart attack", limit=20)
    codes = {r.code for r in results}
    assert "I21.9" in codes or "I22.9" in codes, f"'heart attack' must resolve to MI family; got {codes}"


def test_synonym_layer_canonical_returns_only_listed_codes():
    """A direct canonical hit must return ONLY the curated code list,
    not also substring-matched extras. This lets us reason about cohort
    queries deterministically."""
    results = search(vocabulary="icd10", query="atrial fibrillation", limit=50)
    codes = {r.code for r in results}
    # Curated AFib codes per synonyms.json.
    assert codes == {"I48.0", "I48.91"}, f"got {codes}"


def test_substring_fallback_unmatched_term():
    """A query with no synonym hit should fall back to substring search."""
    results = search(vocabulary="icd10", query="appendicitis", limit=5)
    assert any("appendicitis" in r.display_name.lower() for r in results)


def test_synonym_layer_skips_phantom_codes():
    """If the synonym map references a code that isn't in the codes data,
    the returned list should silently omit it (no crash, no fabricated
    code). Smoke test across the curated top conditions to confirm map
    and data stay in sync."""
    for query in ("htn", "dm", "chf", "afib", "cad", "ckd", "asthma", "copd", "mi", "stroke"):
        results = search(vocabulary="icd10", query=query, limit=20)
        assert len(results) >= 1, f"{query!r} returned no codes — synonym map and icd10 data out of sync?"


def test_synonym_layer_case_insensitive():
    for variant in ("HTN", "htn", "Htn", "hTn"):
        results = search(vocabulary="icd10", query=variant, limit=5)
        codes = {r.code for r in results}
        assert "I10" in codes, f"case variant {variant!r} should match; got {codes}"


def test_synonym_layer_does_not_affect_other_vocabularies():
    """The synonym map only has icd10 entries today. Other vocabularies
    should behave exactly like before (substring search)."""
    results = search(vocabulary="loinc", query="hemoglobin a1c", limit=5)
    assert any("a1c" in r.display_name.lower() for r in results)
    results = search(vocabulary="cvx", query="mmr", limit=5)
    assert any(r.code == "03" for r in results)
