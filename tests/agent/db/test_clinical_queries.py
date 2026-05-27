from agent.db.analytics_adapter import AnalyticsDbAdapter
from agent.db.queries.clinical import demographics_sql, encounters_sql


def test_demographics_returns_one_row_per_source_id(synthetic_db):
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = demographics_sql(source_id="src-john-1962")
    rows = adapter.fetch_all(sql, params)
    assert len(rows) == 1
    assert rows[0]["first_name"] == "John"


def test_encounters_for_source_id(synthetic_db):
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = encounters_sql(source_id="src-john-1962")
    rows = adapter.fetch_all(sql, params)
    assert len(rows) == 4
    facility_names = {r["facility_name"] for r in rows}
    assert "Buffalo Medical Group" in facility_names
    assert "Buffalo General" in facility_names


def test_encounters_date_range_filter(synthetic_db):
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = encounters_sql(
        source_id="src-john-1962",
        start_date="2026-03-01",
        end_date="2026-04-30",
    )
    rows = adapter.fetch_all(sql, params)
    assert len(rows) == 1


def test_encounters_facility_type_inpatient(synthetic_db):
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = encounters_sql(source_id="src-john-1971", facility_type="inpatient")
    rows = adapter.fetch_all(sql, params)
    # place_of_service '21' = inpatient hospital
    assert len(rows) == 1
    assert rows[0]["place_of_service"] == "21"
