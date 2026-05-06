from agent.db.analytics_adapter import AnalyticsDbAdapter
from agent.db.queries.medications import medications_sql


def test_medications_for_source_id(synthetic_db):
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = medications_sql(source_id="src-john-1962")
    rows = adapter.fetch_all(sql, params)
    assert len(rows) == 2


def test_medications_filtered_by_rxnorm(synthetic_db):
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = medications_sql(
        source_id="src-john-1962",
        rxnorm_codes=["6809"],
    )
    rows = adapter.fetch_all(sql, params)
    assert len(rows) == 1
    assert rows[0]["med_name"] == "Metformin"


def test_medications_filtered_by_date_range(synthetic_db):
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = medications_sql(
        source_id="src-john-1962",
        start_date="2026-04-01",
    )
    rows = adapter.fetch_all(sql, params)
    assert len(rows) == 1
    assert rows[0]["med_name"] == "Azithromycin"
