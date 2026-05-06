from agent.db.analytics_adapter import AnalyticsDbAdapter
from agent.db.queries.immunizations import immunizations_sql


def test_immunizations_for_source_id(synthetic_db):
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = immunizations_sql(source_id="src-john-1962")
    rows = adapter.fetch_all(sql, params)
    assert len(rows) == 2


def test_immunizations_filtered_by_cvx(synthetic_db):
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = immunizations_sql(
        source_id="src-john-1962",
        cvx_codes=["03"],
    )
    rows = adapter.fetch_all(sql, params)
    assert len(rows) == 1
    assert rows[0]["vaccine"].startswith("Measles")


def test_immunizations_filtered_by_text(synthetic_db):
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = immunizations_sql(
        source_id="src-john-1962",
        vaccine_text="tdap",
    )
    rows = adapter.fetch_all(sql, params)
    assert len(rows) == 1


def test_immunizations_filtered_by_date_range(synthetic_db):
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = immunizations_sql(
        source_id="src-john-1962",
        start_date="2010-01-01",
    )
    rows = adapter.fetch_all(sql, params)
    assert len(rows) == 1
