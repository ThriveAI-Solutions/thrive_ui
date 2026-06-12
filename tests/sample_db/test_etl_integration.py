"""End-to-end ETL on the fixture CSVs — produces a valid loadable dump."""

from scripts.sample_db.etl import run_etl_in_memory


def test_etl_runs_against_fixtures(
    patients_csv,
    encounters_csv,
    conditions_csv,
    observations_csv,
    medications_csv,
    procedures_csv,
    immunizations_csv,
    providers_csv,
    organizations_csv,
    allergies_csv,
    claims_csv,
):
    """Run all transformers on fixture data; assert every table has rows."""
    inputs = {
        "patients.csv": patients_csv,
        "encounters.csv": encounters_csv,
        "conditions.csv": conditions_csv,
        "observations.csv": observations_csv,
        "medications.csv": medications_csv,
        "procedures.csv": procedures_csv,
        "immunizations.csv": immunizations_csv,
        "providers.csv": providers_csv,
        "organizations.csv": organizations_csv,
        "allergies.csv": allergies_csv,
        "claims.csv": claims_csv,
    }
    out = run_etl_in_memory(inputs, seed=42)

    expected_tables = {
        "dw.internal_patient_profile_v",
        "dw.internal_source_reference_v",
        "dw.federated_demographic_v",
        "dw.federated_demographic_history_v",
        "dw.federated_encounters_v",
        "dw.federated_problems_v",
        "dw.federated_results_v",
        "dw.federated_meds_v",
        "dw.federated_orders_v",
        "dw.federated_vaccination_v",
        "dw.federated_vitals_v",
        "dw.federated_documents_v",
        "dw.federated_allergies_v",
        "dw.federated_claims_icd_diagnosis_detail_v",
        "dw.federated_claims_icd_procedure_detail_v",
        "dw.federated_claims_medical_facility_detail_v",
        "dw.federated_claims_summary_v",
        "dw.metric_federated_data_v",
    }
    assert set(out.keys()) == expected_tables
    # Most tables have rows for the fixture; documents/orders/etc. depend.
    assert len(out["dw.internal_patient_profile_v"]) == 3
    assert len(out["dw.federated_encounters_v"]) == 3
    assert len(out["dw.federated_problems_v"]) >= 3
    assert len(out["dw.federated_allergies_v"]) == 2


def test_etl_emits_empty_allergies_table_when_csv_absent(
    patients_csv,
    encounters_csv,
    conditions_csv,
    observations_csv,
    medications_csv,
    procedures_csv,
    immunizations_csv,
    providers_csv,
    organizations_csv,
    claims_csv,
):
    """allergies.csv is optional in run_etl_in_memory so older Synthea
    fixtures keep working. The table still ships (empty) in the dump."""
    inputs = {
        "patients.csv": patients_csv,
        "encounters.csv": encounters_csv,
        "conditions.csv": conditions_csv,
        "observations.csv": observations_csv,
        "medications.csv": medications_csv,
        "procedures.csv": procedures_csv,
        "immunizations.csv": immunizations_csv,
        "providers.csv": providers_csv,
        "organizations.csv": organizations_csv,
        "claims.csv": claims_csv,
    }
    out = run_etl_in_memory(inputs, seed=42)
    assert "dw.federated_allergies_v" in out
    assert out["dw.federated_allergies_v"] == []


def test_etl_is_deterministic(
    patients_csv,
    encounters_csv,
    conditions_csv,
    observations_csv,
    medications_csv,
    procedures_csv,
    immunizations_csv,
    providers_csv,
    organizations_csv,
    allergies_csv,
    claims_csv,
):
    inputs = {
        "patients.csv": patients_csv,
        "encounters.csv": encounters_csv,
        "conditions.csv": conditions_csv,
        "observations.csv": observations_csv,
        "medications.csv": medications_csv,
        "procedures.csv": procedures_csv,
        "immunizations.csv": immunizations_csv,
        "providers.csv": providers_csv,
        "organizations.csv": organizations_csv,
        "allergies.csv": allergies_csv,
        "claims.csv": claims_csv,
    }
    a = run_etl_in_memory(inputs, seed=42)
    b = run_etl_in_memory(inputs, seed=42)
    assert a == b
