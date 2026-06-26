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
    assert str(rows[0]["admit_date"]).startswith("2026-03-15 18:00")


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


def test_missing_visit_numbers_do_not_collapse_into_one_stay(synthetic_db):
    sql, params = admissions_sql(source_id="src-anne-1948", dialect="sqlite", facility_type="any")
    rows = _adapter(synthetic_db).fetch_all(sql, params)
    assert len(rows) == 2
    assert {r["visit_number"] for r in rows} == {None}
    assert {str(r["admit_date"]) for r in rows} == {"2026-01-01 08:00", "2026-01-02 09:00"}
    assert all(r["is_inpatient_admission"] == 0 for r in rows)


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


# --- Real-life federated ADT exposure (patient 8 / src-susan-1955) ---
# Some HEALTHeLINK source systems emit ONLY A08/A31 messages, which still
# represent genuine encounters. A visit is exposed when it has any real
# patient-facility contact (any care setting or lifecycle event); only
# pre-admit/pending placeholders and cancelled admits are suppressed.


def test_outpatient_filter_includes_a08_only_radiology_visit(synthetic_db):
    """Windsong-style radiology emits only A08 with an OUTPATIENT setting (V800);
    that IS an outpatient encounter and must surface alongside the
    registration/discharge ambulatory visit (V801)."""
    sql, params = admissions_sql(source_id="src-susan-1955", dialect="sqlite", facility_type="outpatient")
    rows = _adapter(synthetic_db).fetch_all(sql, params)
    assert {r["visit_number"] for r in rows} == {"V800", "V801"}


def test_a08_only_radiology_visit_anchors_on_its_own_event(synthetic_db):
    sql, params = admissions_sql(source_id="src-susan-1955", dialect="sqlite", facility_type="outpatient")
    rows = _adapter(synthetic_db).fetch_all(sql, params)
    v800 = next(r for r in rows if r["visit_number"] == "V800")
    assert v800["event_location"] == "Windsong Radiology"
    assert v800["setting"] == "OUTPATIENT"
    assert str(v800["admit_date"]).startswith("2026-05-01 08:00")


def test_outpatient_visit_anchors_on_registration_not_admin_update(synthetic_db):
    """V801 has an earlier admin A08 (08:00) and a REGISTRATION (09:00); the
    visit start and admitting location come from the lifecycle event, not the
    admin update."""
    sql, params = admissions_sql(source_id="src-susan-1955", dialect="sqlite", facility_type="outpatient")
    rows = _adapter(synthetic_db).fetch_all(sql, params)
    v801 = next(r for r in rows if r["visit_number"] == "V801")
    assert v801["event_location"] == "Kaleida Ambulatory"
    assert v801["location_type"] == "Clinic"
    assert str(v801["admit_date"]).startswith("2026-05-02 09:00")
    assert str(v801["discharge_date"]).startswith("2026-05-02 10:00")


def test_any_surfaces_unknown_setting_contact_and_excludes_preadmit(synthetic_db):
    """'any' shows every real patient-facility contact, including the
    unknown-setting A31 practice (V802), but never a pre-admit-only placeholder
    (V803). None of patient 8's contacts are inpatient."""
    sql, params = admissions_sql(source_id="src-susan-1955", dialect="sqlite", facility_type="any")
    rows = _adapter(synthetic_db).fetch_all(sql, params)
    assert {r["visit_number"] for r in rows} == {"V800", "V801", "V802"}
    assert all(r["is_inpatient_admission"] == 0 for r in rows)


def test_unknown_setting_contact_excluded_from_outpatient_filter(synthetic_db):
    """V802 (Unknown setting) is a real contact but is not classified OUTPATIENT,
    so facility_type='outpatient' must not return it."""
    sql, params = admissions_sql(source_id="src-susan-1955", dialect="sqlite", facility_type="outpatient")
    rows = _adapter(synthetic_db).fetch_all(sql, params)
    assert "V802" not in {r["visit_number"] for r in rows}


def test_preadmit_only_outpatient_is_not_a_visit(synthetic_db):
    """V803 is only a 'P' preadmit-class A08 plus an A05 pending admit — not a
    completed encounter, so it never surfaces."""
    sql, params = admissions_sql(source_id="src-susan-1955", dialect="sqlite", facility_type="any")
    rows = _adapter(synthetic_db).fetch_all(sql, params)
    assert "V803" not in {r["visit_number"] for r in rows}


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
    assert "CAST(isr.patient_id AS VARCHAR) = adt.patient_id" in n
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
    assert str(rows[0]["admit_date"]).startswith("2026-03-15 18:00")
