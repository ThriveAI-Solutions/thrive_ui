from agent.system_prompt import SYSTEM_PROMPT


def test_prompt_mentions_phase2_tools():
    for tool in (
        "find_patient",
        "get_patient_clinical_data",
        "list_patient_documents",
        "search_codes",
        "search_knowledge_base",
    ):
        assert tool in SYSTEM_PROMPT


def test_prompt_mentions_per_domain_guidance():
    for domain in ("labs", "diagnoses", "medications", "immunizations", "procedures", "imaging", "encounters"):
        assert domain in SYSTEM_PROMPT.lower()


def test_prompt_warns_about_impressions_and_note_bodies():
    p = SYSTEM_PROMPT.lower()
    assert "impression" in p
    assert "note bod" in p


def test_prompt_warns_about_loinc_coverage():
    assert "loinc" in SYSTEM_PROMPT.lower()


def test_prompt_explains_search_codes_use():
    p = SYSTEM_PROMPT.lower()
    assert "search_codes" in SYSTEM_PROMPT
    assert "cvx" in p or "rxnorm" in p
