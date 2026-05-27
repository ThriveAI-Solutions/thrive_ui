"""Offline lint over the representative_questions.yaml suite.

This test does NOT call any LLM. It validates the YAML is well-formed
and that every question references tools and domains the agent code
actually exposes. The live-model regression runs in
scripts/run_representative_regression.py and is gated separately.
"""

from pathlib import Path
import yaml


_YAML = Path(__file__).parent / "representative_questions.yaml"


def test_yaml_loads_and_has_ten_questions():
    data = yaml.safe_load(_YAML.read_text())
    assert isinstance(data, dict)
    assert "questions" in data
    assert len(data["questions"]) == 10


def test_each_question_has_required_fields():
    data = yaml.safe_load(_YAML.read_text())
    required = {"id", "prompt", "expected_tools", "expected_data_availability", "must_not_contain"}
    for q in data["questions"]:
        missing = required - set(q.keys())
        assert not missing, f"question {q.get('id')!r} missing fields: {missing}"


def test_expected_tools_reference_real_tools():
    data = yaml.safe_load(_YAML.read_text())
    valid = {
        "find_patient",
        "get_patient_clinical_data",
        "list_patient_documents",
        "search_codes",
        "search_knowledge_base",
    }
    for q in data["questions"]:
        for t in q["expected_tools"]:
            assert t in valid, f"unknown tool {t!r} in question {q['id']}"


def test_expected_domains_reference_real_domains():
    data = yaml.safe_load(_YAML.read_text())
    valid_domains = {
        "demographics",
        "encounters",
        "labs",
        "diagnoses",
        "medications",
        "immunizations",
        "procedures",
        "imaging",
    }
    for q in data["questions"]:
        for d in q.get("expected_domains", []):
            assert d in valid_domains, f"unknown domain {d!r} in question {q['id']}"


def test_q10_marked_deferred():
    data = yaml.safe_load(_YAML.read_text())
    q10 = next(q for q in data["questions"] if q["id"] == "Q10")
    assert q10.get("deferred") is True
    assert q10["expected_data_availability"] == "domain_not_available"
