from scripts.sample_db.crosswalks.loader import rxnorm_to_ndc, snomed_to_icd10


def test_snomed_diabetes_maps_to_icd10_e11_9():
    assert snomed_to_icd10()["44054006"] == "E11.9"


def test_snomed_hypertension_maps_to_i10():
    assert snomed_to_icd10()["38341003"] == "I10"


def test_rxnorm_metformin_maps_to_ndc():
    ndc = rxnorm_to_ndc()["6809"]
    # NDC format: 5-4-2 with separators
    assert len(ndc.split("-")) == 3


def test_unmapped_returns_no_entry():
    assert "0000000" not in snomed_to_icd10()


def test_crosswalks_have_minimum_coverage():
    # Sanity bar — adjust if seed grows.
    assert len(snomed_to_icd10()) >= 40
    assert len(rxnorm_to_ndc()) >= 40
