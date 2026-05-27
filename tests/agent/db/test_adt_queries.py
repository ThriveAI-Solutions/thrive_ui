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


def test_admissions_filtered_by_emergency(synthetic_db):
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = admissions_sql(source_id="src-john-1962", facility_type="emergency")
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
