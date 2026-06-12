# tests/agent/codes/test_loader.py
import pytest
from agent.codes.loader import VocabLoader, search


def test_loader_loads_all_vocabularies():
    loader = VocabLoader()
    for vocab in ("loinc", "cvx", "rxnorm", "icd10", "cpt", "snomed"):
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
    """Substring search must still work in any vocabulary that doesn't
    have a synonym map entry for the given query."""
    results = search(vocabulary="loinc", query="hemoglobin a1c", limit=5)
    assert any("a1c" in r.display_name.lower() for r in results)
    results = search(vocabulary="cvx", query="mmr", limit=5)
    assert any(r.code == "03" for r in results)


# ---------------------------------------------------------------------------
# SNOMED allergy intent shortcuts (Epic #203)
# ---------------------------------------------------------------------------


def test_snomed_penicillin_allergy_intent():
    """'penicillin allergy' must hit the curated drug-allergy bucket and
    return the penicillin + amoxicillin (cross-reactive) codes."""
    results = search(vocabulary="snomed", query="penicillin allergy", limit=10)
    codes = {r.code for r in results}
    assert "91936005" in codes, f"expected 91936005 in {codes}"
    assert "294505008" in codes, f"expected 294505008 (cross-reactive amoxicillin) in {codes}"


def test_snomed_food_allergy_intent_returns_full_food_bucket():
    """'any food allergy' must resolve to the curated food-allergy bucket."""
    results = search(vocabulary="snomed", query="any food allergy", limit=20)
    codes = {r.code for r in results}
    expected = {"91934008", "91935009", "300913006", "417532002", "91938006", "91930004", "300915002"}
    assert expected.issubset(codes), f"missing food allergy codes: {expected - codes}"


def test_snomed_peanut_allergy_lay_term():
    """'peanut allergy' resolves to the single peanut SNOMED code."""
    results = search(vocabulary="snomed", query="peanut allergy", limit=5)
    codes = {r.code for r in results}
    assert codes == {"91934008"}, f"got {codes}"


def test_snomed_hay_fever_resolves_to_environmental_bucket():
    """Lay term 'hay fever' must reach environmental allergy codes."""
    results = search(vocabulary="snomed", query="hay fever", limit=10)
    codes = {r.code for r in results}
    assert "232347008" in codes, f"expected pollen rhinitis 232347008 in {codes}"
    assert any(c in codes for c in ("367498001", "446096008")), (
        f"expected at least one allergic-rhinitis code in {codes}"
    )


def test_snomed_anaphylaxis_substring_fallback():
    """Anaphylaxis is reachable through both the canonical key and via
    plain substring match on the display name."""
    by_canonical = search(vocabulary="snomed", query="anaphylaxis", limit=5)
    assert any(r.code == "39579001" for r in by_canonical)


def test_snomed_sulfa_allergy_synonym():
    """'sulfa allergy' (lay shorthand) resolves to the sulfonamide code."""
    results = search(vocabulary="snomed", query="sulfa allergy", limit=5)
    codes = {r.code for r in results}
    assert codes == {"91937001"}, f"got {codes}"


def test_snomed_drug_allergy_intent_returns_full_drug_bucket():
    """'drug allergy' / 'medication allergy' must surface the full
    curated drug-allergy bucket."""
    for needle in ("drug allergy", "medication allergy", "any drug allergy"):
        results = search(vocabulary="snomed", query=needle, limit=10)
        codes = {r.code for r in results}
        expected = {"91936005", "294505008", "91937001", "293584003", "293586001"}
        assert expected.issubset(codes), f"{needle!r} missing codes: {expected - codes}"


def test_snomed_synonyms_codes_all_exist_in_data():
    """Drift guard: every code listed in the synonyms.json snomed block
    must exist in agent/codes/data/snomed.json. Catches typos before they
    silently fall through the loader's phantom-code filter."""
    from agent.codes.loader import _load, _load_synonyms

    data_codes = {r["code"] for r in _load("snomed")}
    snomed_block = _load_synonyms().get("snomed", {})
    for canonical, entry in snomed_block.items():
        for code in entry.get("codes", []):
            assert code in data_codes, (
                f"synonyms.json snomed[{canonical!r}] references {code!r} which is not in snomed.json"
            )
