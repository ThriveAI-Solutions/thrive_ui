from agent.db.analytics_adapter import AnalyticsDbAdapter
from agent.db.queries.adt import admissions_sql


def test_admissions_for_source_id(synthetic_db):
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = admissions_sql(source_id="src-john-1962")
    rows = adapter.fetch_all(sql, params)
    assert len(rows) == 4
    # Should be ordered by event_date DESC
    dates = [r["event_date"] for r in rows]
    assert dates == sorted(dates, reverse=True)


def test_admissions_filtered_by_inpatient(synthetic_db):
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = admissions_sql(source_id="src-john-1962", facility_type="inpatient")
    rows = adapter.fetch_all(sql, params)
    assert len(rows) == 2
    assert all(r["setting"] == "INPATIENT" for r in rows)


def test_admissions_filtered_by_snf(synthetic_db):
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = admissions_sql(source_id="src-john-1962", facility_type="snf")
    rows = adapter.fetch_all(sql, params)
    assert len(rows) == 1
    assert rows[0]["event_location"] == "Sunrise SNF"


def test_admissions_filtered_by_ltc_includes_snf(synthetic_db):
    """ltc and snf facility types should match the same SNF rows."""
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = admissions_sql(source_id="src-john-1962", facility_type="ltc")
    rows = adapter.fetch_all(sql, params)
    assert len(rows) == 1
    assert rows[0]["setting"] == "SNF"


def test_admissions_filtered_by_ed(synthetic_db):
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = admissions_sql(source_id="src-john-1962", facility_type="ed")
    rows = adapter.fetch_all(sql, params)
    assert len(rows) == 1
    assert rows[0]["setting"] == "EMERGENCY"


def test_admissions_filtered_by_date_range(synthetic_db):
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = admissions_sql(
        source_id="src-john-1962",
        start_date="2025-06-01",
        end_date="2025-06-30",
    )
    rows = adapter.fetch_all(sql, params)
    assert len(rows) == 2


def test_admissions_without_discharge_details(synthetic_db):
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = admissions_sql(
        source_id="src-john-1962",
        include_discharge_details=False,
    )
    rows = adapter.fetch_all(sql, params)
    assert len(rows) == 4
    # Discharge columns should not be in the result
    assert "discharge_disposition" not in rows[0]
    assert "discharge_location" not in rows[0]


def test_admissions_no_records_for_unknown_source(synthetic_db):
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = admissions_sql(source_id="src-nonexistent")
    rows = adapter.fetch_all(sql, params)
    assert len(rows) == 0


def test_admissions_facility_type_any(synthetic_db):
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = admissions_sql(source_id="src-john-1962", facility_type="any")
    rows = adapter.fetch_all(sql, params)
    assert len(rows) == 4


def test_admissions_resolves_source_id_via_isr_join():
    """Regression for #196: production federated_adt_v has no `source_id`
    column — only `patient_id`. The agent's identity model is source_id-based,
    so admissions_sql must JOIN internal_source_reference_v at empi_rank=1
    instead of filtering federated_adt_v directly on a non-existent column."""
    sql, _ = admissions_sql(source_id="any-cid", schema_prefix="dw.")
    normalized = " ".join(sql.split())
    # Must NOT filter federated_adt_v on source_id (the column doesn't exist).
    assert "WHERE source_id" not in normalized
    assert "adt.source_id" not in normalized
    # Must JOIN through internal_source_reference_v at empi_rank=1.
    assert "JOIN dw.internal_source_reference_v isr" in normalized
    assert "isr.patient_id = adt.patient_id" in normalized
    assert "isr.empi_rank = 1" in normalized
    assert "isr.source_id = :source_id" in normalized
    # The result must still echo source_id into each row so the tool layer
    # can populate AdmissionItem.source_id.
    assert "isr.source_id AS source_id" in normalized


def test_admissions_source_id_returned_in_rows(synthetic_db):
    """The SELECT echoes the input source_id back into each row so the
    tool layer's row-shaping code stays uniform across domains."""
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = admissions_sql(source_id="src-john-1962")
    rows = adapter.fetch_all(sql, params)
    assert rows, "expected admissions rows for src-john-1962"
    assert all(r["source_id"] == "src-john-1962" for r in rows)


def test_admissions_does_not_leak_other_patients(synthetic_db):
    """Defense check: the JOIN must scope rows to the requested source_id.
    src-john-1971 (patient_id=2) has exactly one ADT row in the fixture;
    it must not contaminate src-john-1962's (patient_id=1) result set."""
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = admissions_sql(source_id="src-john-1971")
    rows = adapter.fetch_all(sql, params)
    assert len(rows) == 1
    assert rows[0]["event_location"] == "Kaleida Methodist"
