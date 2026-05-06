from sqlalchemy import text


def test_three_patients_present(synthetic_db):
    with synthetic_db.connect() as conn:
        n = conn.execute(text("SELECT COUNT(*) FROM internal_patient_profile_v")).scalar()
    assert n == 3


def test_inactive_rank99_present_but_will_be_filtered(synthetic_db):
    with synthetic_db.connect() as conn:
        n = conn.execute(text("SELECT COUNT(*) FROM internal_source_reference_v WHERE empi_rank = 99")).scalar()
    assert n == 1
