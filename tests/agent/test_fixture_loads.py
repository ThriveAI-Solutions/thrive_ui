from sqlalchemy import text


def test_three_patients_present(synthetic_db):
    with synthetic_db.connect() as conn:
        n = conn.execute(text("SELECT COUNT(*) FROM internal_patient_profile_v")).scalar()
    assert n >= 3


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


def test_synthetic_db_has_metric_federated_data_rows(synthetic_db):
    from sqlalchemy import text

    with synthetic_db.connect() as conn:
        n = conn.execute(text("SELECT COUNT(*) FROM metric_federated_data_v")).scalar_one()
    assert n >= 8, f"expected ≥8 metric rows for cohort coverage, got {n}"


def test_synthetic_db_has_diabetic_cohort_at_kaleida(synthetic_db):
    """Phase 4 cohort SQL needs ≥2 diabetic patients seen at Kaleida to verify
    facility + diagnosis_codes filters in combination."""
    from sqlalchemy import text

    with synthetic_db.connect() as conn:
        rows = conn.execute(
            text("""
            SELECT p.patient_id
            FROM internal_patient_profile_v p
            JOIN internal_source_reference_v isr ON isr.patient_id = p.patient_id AND isr.empi_rank != 99
            JOIN metric_federated_data_v m ON m.patient_id = p.patient_id
            WHERE m.code = 'E11.9'
              AND m.code_type = 'ICD-10'
              AND LOWER(p.practice_name) LIKE '%kaleida%'
        """)
        ).fetchall()
    assert len(rows) >= 2, f"expected ≥2 diabetic Kaleida patients, got {len(rows)}"


def test_synthetic_db_has_metformin_patient(synthetic_db):
    from sqlalchemy import text

    with synthetic_db.connect() as conn:
        n = conn.execute(
            text("""
            SELECT COUNT(DISTINCT patient_id)
            FROM metric_federated_data_v
            WHERE code = '6809' AND code_type = 'RxNorm'
        """)
        ).scalar_one()
    assert n >= 2, f"expected ≥2 patients on metformin (RxNorm 6809), got {n}"
