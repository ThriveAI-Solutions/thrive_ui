from agent.codes.match_forms import code_match_forms


def test_dotted_input_carries_undotted_form():
    forms = code_match_forms(["E11.9"])
    assert "E11.9" in forms and "E119" in forms


def test_undotted_icd_gains_dotted_forms():
    forms = code_match_forms(["E119"])
    assert "E119" in forms and "E11.9" in forms


def test_snomed_numeric_passes_through_untouched():
    assert code_match_forms(["44054006"]) == ["44054006"]


def test_dedup_and_order_preserved():
    forms = code_match_forms(["E11.9", "E119"])
    assert len(forms) == len(set(forms))
