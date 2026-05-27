import pytest
from agent.db.analytics_adapter import AnalyticsDbAdapter
from agent.db.queries.patient import find_patient_sql


def test_find_patient_returns_three_unique_smiths(synthetic_db):
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = find_patient_sql(last_name="Smith", limit=25)
    rows = adapter.fetch_all(sql, params)
    source_ids = sorted({r["source_id"] for r in rows})
    assert source_ids == ["src-jane-1985", "src-john-1962", "src-john-1971"]


def test_find_patient_excludes_rank_99(synthetic_db):
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = find_patient_sql(last_name="Smith", limit=25)
    rows = adapter.fetch_all(sql, params)
    inactive_id = "src-john-1962-stale"
    assert all(r["source_id"] != inactive_id for r in rows)


def test_find_patient_filters_by_first_name(synthetic_db):
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = find_patient_sql(first_name="Jane", last_name="Smith", limit=25)
    rows = adapter.fetch_all(sql, params)
    assert len(rows) == 1
    assert rows[0]["first_name"] == "Jane"


def test_find_patient_filters_by_dob(synthetic_db):
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = find_patient_sql(last_name="Smith", dob="1962-05-01", limit=25)
    rows = adapter.fetch_all(sql, params)
    assert len(rows) == 1
    assert rows[0]["source_id"] == "src-john-1962"


def test_find_patient_sql_rejects_empty_criteria():
    """Defense in depth — direct callers must not be able to issue an
    unfiltered scan of the patient table."""
    with pytest.raises(ValueError):
        find_patient_sql(limit=25)
