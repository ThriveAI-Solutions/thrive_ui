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


def test_system_prompt_documents_run_sql():
    from agent.system_prompt import SYSTEM_PROMPT

    assert "run_sql" in SYSTEM_PROMPT
    # The prompt should make clear run_sql is a fallback for when the
    # curated tools cannot answer (Phase 3 design §3.4).
    assert "escape hatch" in SYSTEM_PROMPT.lower() or "fallback" in SYSTEM_PROMPT.lower()


def test_system_prompt_documents_make_chart():
    from agent.system_prompt import SYSTEM_PROMPT

    assert "make_chart" in SYSTEM_PROMPT


def test_system_prompt_documents_summarize_results():
    from agent.system_prompt import SYSTEM_PROMPT

    assert "summarize_results" in SYSTEM_PROMPT


def test_system_prompt_documents_search_patients_by_criteria():
    from agent.system_prompt import SYSTEM_PROMPT

    assert "search_patients_by_criteria" in SYSTEM_PROMPT
    # The prompt must teach the population-vs-specific-patient routing rule
    # (Phase 4 design §3.3). Look for either phrase, since wording can evolve.
    assert "population" in SYSTEM_PROMPT.lower() or "cohort" in SYSTEM_PROMPT.lower()
