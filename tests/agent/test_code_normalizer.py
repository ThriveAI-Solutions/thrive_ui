import pytest
from agent.code_normalizer import variants_for, normalize_token


def test_variants_for_loinc_includes_common_spellings():
    v = variants_for("loinc")
    assert "LOINC" in v
    assert "loinc" in v
    assert "LN" in v


def test_variants_for_icd10_includes_dot_and_dash():
    v = variants_for("icd10")
    assert "ICD-10" in v
    assert "ICD10" in v
    assert "ICD-10-CM" in v


def test_variants_for_cvx():
    assert "CVX" in variants_for("cvx")


def test_variants_for_rxnorm():
    assert "RXNORM" in variants_for("rxnorm")
    assert "RxNorm" in variants_for("rxnorm")


def test_variants_for_cpt():
    assert "CPT" in variants_for("cpt")
    assert "CPT4" in variants_for("cpt")


def test_unknown_token_raises():
    with pytest.raises(ValueError, match="unknown vocabulary"):
        variants_for("snomed-fake")


def test_normalize_token_collapses_to_canonical():
    assert normalize_token("LOINC") == "loinc"
    assert normalize_token("LN") == "loinc"
    assert normalize_token("ICD-10-CM") == "icd10"
    assert normalize_token("RxNorm") == "rxnorm"


def test_normalize_token_unknown_returns_none():
    assert normalize_token("zzz-not-real") is None
