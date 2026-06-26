from sqlalchemy import text

from agent.db.queries.cohort import cohort_sql


def _counts(engine, criteria):
    sql, params = cohort_sql(criteria, schema_prefix="")
    with engine.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()
    return rows


class _C:
    diagnosis_codes = None
    diagnosis_date_range = None
    medication_rxnorm_codes = None
    condition_text = None
    age_min = None
    age_max = None
    gender = None
    facility = None
    last_visit_after = None
    last_visit_before = None
    zip_code = None
    city = None
    state = None
    inpatient_admission = None
    inpatient_admission_date_range = None
    sample_size = 0


def _make(**kw):
    c = _C()
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def test_inpatient_cohort_count(synthetic_db):
    # qualifying inpatient patients in the fixture: p1 (V100), p2 (V200), p4 (V400/V401) = 3
    rows = _counts(synthetic_db, _make(inpatient_admission=True))
    assert rows[0]._mapping["total_count"] == 3


def test_inpatient_cohort_casts_profile_patient_id_for_adt_join():
    sql, _ = cohort_sql(_make(inpatient_admission=True), schema_prefix="dw.", dialect="redshift")
    n = " ".join(sql.split())
    assert "adt_ip.patient_id = CAST(p.patient_id AS VARCHAR)" in n


def test_inpatient_cohort_dedups_multiple_stays(synthetic_db):
    # p4 has two qualifying stays but must count once.
    rows = _counts(synthetic_db, _make(inpatient_admission=True, sample_size=20))
    source_ids = {r._mapping["source_id"] for r in rows}
    assert source_ids == {"src-john-1962", "src-john-1971", "src-mary-1956"}


def test_inpatient_cohort_date_window(synthetic_db):
    from types import SimpleNamespace

    dr = SimpleNamespace(start=__import__("datetime").date(2026, 1, 1), end=__import__("datetime").date(2026, 12, 31))
    rows = _counts(synthetic_db, _make(inpatient_admission=True, inpatient_admission_date_range=dr))
    # qualifying admit date in 2026: p2 (2026-03), p4/V400 (2026-03). p1/V100 is 2025. = 2
    assert rows[0]._mapping["total_count"] == 2
