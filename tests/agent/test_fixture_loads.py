from sqlalchemy import text


def test_three_patients_present(synthetic_db):
    with synthetic_db.connect() as conn:
        n = conn.execute(text("SELECT COUNT(*) FROM internal_patient_profile_v")).scalar()
    assert n == 3


def test_inactive_rank99_present_but_will_be_filtered(synthetic_db):
    with synthetic_db.connect() as conn:
        n = conn.execute(text("SELECT COUNT(*) FROM internal_source_reference_v WHERE empi_rank = 99")).scalar()
    assert n == 1


def test_synthetic_db_has_phase2_domain_rows(synthetic_db):
    from sqlalchemy import text

    with synthetic_db.connect() as conn:
        for view in (
            "federated_results_v",
            "federated_problems_v",
            "federated_meds_v",
            "federated_orders_v",
            "federated_vaccination_v",
            "federated_documents_v",
        ):
            count = conn.execute(text(f"SELECT COUNT(*) FROM {view} WHERE source_id = 'src-john-1962'")).scalar()
            assert count >= 1, f"{view} has no fixture rows for src-john-1962"

        # federated_claims_icd_procedure_detail_v has no source_id column per
        # redshift_tables.json (Task 4 reconciliation). Check total row count instead.
        claims_count = conn.execute(text("SELECT COUNT(*) FROM federated_claims_icd_procedure_detail_v")).scalar()
        assert claims_count >= 1
