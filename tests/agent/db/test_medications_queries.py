from agent.db.analytics_adapter import AnalyticsDbAdapter
from agent.db.queries.medications import medications_sql


def test_medications_returns_full_list_with_status(synthetic_db):
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = medications_sql(source_id="src-john-1962")
    rows = adapter.fetch_all(sql, params)
    assert len(rows) == 2
    by_name = {r["med_name"]: r for r in rows}
    assert by_name["Metformin"]["status"] == "active"
    assert by_name["Metformin"]["date_stopped"] is None
    assert by_name["Azithromycin"]["status"] == "completed"


def test_medications_sql_has_no_rxnorm_filter():
    sql, params = medications_sql(source_id="src-john-1962")
    assert "rxnorm_code IN" not in sql
    assert not any(k.startswith("rx_") for k in params)


def test_medications_filtered_by_date_range(synthetic_db):
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = medications_sql(
        source_id="src-john-1962",
        start_date="2026-04-01",
    )
    rows = adapter.fetch_all(sql, params)
    assert len(rows) == 1
    assert rows[0]["med_name"] == "Azithromycin"
