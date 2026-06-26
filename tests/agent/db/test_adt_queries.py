from agent.db.analytics_adapter import AnalyticsDbAdapter
from agent.db.queries.adt import admissions_sql


def _adapter(engine):
    return AnalyticsDbAdapter(engine=engine, dialect="sqlite")


def test_inpatient_filter_returns_one_stay_per_visit(synthetic_db):
    """p1 has one qualifying inpatient stay (V100), rolled up to a single row."""
    sql, params = admissions_sql(source_id="src-john-1962", dialect="sqlite", facility_type="inpatient")
    rows = _adapter(synthetic_db).fetch_all(sql, params)
    assert len(rows) == 1
    r = rows[0]
    assert r["visit_number"] == "V100"
    assert r["is_inpatient_admission"] == 1
    assert str(r["admit_date"]).startswith("2025-06-15")
    assert str(r["discharge_date"]).startswith("2025-06-18")


def test_any_returns_all_stays_rolled_up(synthetic_db):
    """p1 has three visits (V100 inpatient, V101 ED, V102 SNF)."""
    sql, params = admissions_sql(source_id="src-john-1962", dialect="sqlite", facility_type="any")
    rows = _adapter(synthetic_db).fetch_all(sql, params)
    assert len(rows) == 3
    visits = {r["visit_number"] for r in rows}
    assert visits == {"V100", "V101", "V102"}
    # ordered by admit_date DESC
    dates = [str(r["admit_date"]) for r in rows]
    assert dates == sorted(dates, reverse=True)


def test_ed_only_visit_is_not_inpatient(synthetic_db):
    sql, params = admissions_sql(source_id="src-john-1962", dialect="sqlite", facility_type="any")
    rows = _adapter(synthetic_db).fetch_all(sql, params)
    v101 = next(r for r in rows if r["visit_number"] == "V101")
    assert v101["is_inpatient_admission"] == 0


def test_a06_conversion_counts_as_inpatient(synthetic_db):
    sql, params = admissions_sql(source_id="src-john-1971", dialect="sqlite", facility_type="inpatient")
    rows = _adapter(synthetic_db).fetch_all(sql, params)
    assert len(rows) == 1
    assert rows[0]["visit_number"] == "V200"
    # qualifying date anchors on the A06 row (18:00), not the EMERGENCY registration (12:00)
    assert str(rows[0]["admit_date"]).startswith("2026-03-15")


def test_cancelled_admit_is_not_inpatient(synthetic_db):
    sql, params = admissions_sql(source_id="src-jane-1985", dialect="sqlite", facility_type="inpatient")
    rows = _adapter(synthetic_db).fetch_all(sql, params)
    assert rows == []


def test_preadmit_only_is_not_inpatient(synthetic_db):
    sql, params = admissions_sql(source_id="src-robert-1970", dialect="sqlite", facility_type="inpatient")
    rows = _adapter(synthetic_db).fetch_all(sql, params)
    assert rows == []


def test_junk_settings_are_not_inpatient(synthetic_db):
    sql, params = admissions_sql(source_id="src-daniel-1977", dialect="sqlite", facility_type="inpatient")
    rows = _adapter(synthetic_db).fetch_all(sql, params)
    assert rows == []


def test_two_inpatient_stays_for_one_patient(synthetic_db):
    sql, params = admissions_sql(source_id="src-mary-1956", dialect="sqlite", facility_type="inpatient")
    rows = _adapter(synthetic_db).fetch_all(sql, params)
    assert len(rows) == 2
    assert {r["visit_number"] for r in rows} == {"V400", "V401"}


def test_inpatient_date_range_filters_on_qualifying_admit_date(synthetic_db):
    sql, params = admissions_sql(
        source_id="src-mary-1956",
        dialect="sqlite",
        facility_type="inpatient",
        start_date="2026-01-01",
        end_date="2026-12-31",
    )
    rows = _adapter(synthetic_db).fetch_all(sql, params)
    assert len(rows) == 1
    assert rows[0]["visit_number"] == "V400"


def test_facility_type_snf(synthetic_db):
    sql, params = admissions_sql(source_id="src-john-1962", dialect="sqlite", facility_type="snf")
    rows = _adapter(synthetic_db).fetch_all(sql, params)
    assert len(rows) == 1
    assert rows[0]["visit_number"] == "V102"


def test_facility_type_ed(synthetic_db):
    sql, params = admissions_sql(source_id="src-john-1962", dialect="sqlite", facility_type="ed")
    rows = _adapter(synthetic_db).fetch_all(sql, params)
    assert len(rows) == 1
    assert rows[0]["visit_number"] == "V101"


def test_no_records_for_unknown_source(synthetic_db):
    sql, params = admissions_sql(source_id="src-nonexistent", dialect="sqlite")
    rows = _adapter(synthetic_db).fetch_all(sql, params)
    assert rows == []


def test_resolves_source_id_via_isr_join():
    """Regression #196: federated_adt_v has no source_id column."""
    sql, _ = admissions_sql(source_id="any-cid", dialect="redshift", schema_prefix="dw.")
    n = " ".join(sql.split())
    assert "WHERE source_id" not in n
    assert "adt.source_id" not in n
    assert "JOIN dw.internal_source_reference_v isr" in n
    assert "isr.patient_id = adt.patient_id" in n
    assert "isr.empi_rank = 1" in n
    assert "isr.source_id = :source_id" in n
    assert "isr.source_id AS source_id" in n


def test_does_not_leak_other_patients(synthetic_db):
    sql, params = admissions_sql(source_id="src-john-1971", dialect="sqlite", facility_type="any")
    rows = _adapter(synthetic_db).fetch_all(sql, params)
    assert all(r["source_id"] == "src-john-1971" for r in rows)
    assert {r["visit_number"] for r in rows} == {"V200"}


def test_event_location_is_admitting_facility_on_transfer(synthetic_db):
    """V200 is an ED->inpatient cross-facility transfer: ED registration at
    'Sisters of Charity ED', then admitted INPATIENT (A06) at 'Kaleida
    Methodist'. event_location must be the ADMITTING (qualifying-event)
    facility, not a blind MAX across the visit (which would pick the
    lexicographically-greater 'Sisters of Charity ED')."""
    sql, params = admissions_sql(source_id="src-john-1971", dialect="sqlite", facility_type="inpatient")
    rows = _adapter(synthetic_db).fetch_all(sql, params)
    assert len(rows) == 1
    assert rows[0]["visit_number"] == "V200"
    assert rows[0]["event_location"] == "Kaleida Methodist"
    assert rows[0]["location_type"] == "Hospital"
