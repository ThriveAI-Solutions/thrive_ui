from agent.rag.seed import SCHEMA_DOCS, IDENTITY_DOCS, FRESHNESS_DOCS


def test_schema_docs_cover_whitelist():
    """Per spec §7.12 — every whitelist view must have a schema doc."""
    expected = {
        "internal_patient_profile_v",
        "internal_source_reference_v",
        "federated_demographic_v",
        "federated_demographic_history_v",
        "federated_encounters_v",
        "federated_problems_v",
        "federated_results_v",
        "federated_meds_v",
        "federated_orders_v",
        "federated_vaccination_v",
        "federated_vitals_v",
        "federated_documents_v",
        "metric_federated_data_v",
    }
    covered = {d["view"] for d in SCHEMA_DOCS}
    missing = expected - covered
    assert not missing, f"Missing schema docs for: {missing}"


def test_identity_docs_present():
    assert any("source_id" in d["text"] and "empi_rank" in d["text"] for d in IDENTITY_DOCS)


def test_freshness_docs_present():
    assert any("bi-weekly" in d["text"].lower() for d in FRESHNESS_DOCS)
    assert any("monthly" in d["text"].lower() for d in FRESHNESS_DOCS)
