from agent.db.analytics_adapter import AnalyticsDbAdapter
from agent.db.queries.procedures import procedures_sql


def test_procedures_unions_three_sources(synthetic_db):
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = procedures_sql(source_id="src-john-1962")
    rows = adapter.fetch_all(sql, params)
    sources = {r["source"] for r in rows}
    assert sources == {"orders", "problems", "claims"}


def test_procedures_orders_carry_cpt_codes(synthetic_db):
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = procedures_sql(source_id="src-john-1962")
    rows = adapter.fetch_all(sql, params)
    cpt_codes = {r["code"] for r in rows if r["source"] == "orders" and r["code_type"] == "CPT"}
    assert "45378" in cpt_codes
    assert "71046" in cpt_codes


def test_procedures_filtered_by_cpt(synthetic_db):
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = procedures_sql(
        source_id="src-john-1962",
        cpt_codes=["45378"],
    )
    rows = adapter.fetch_all(sql, params)
    assert len(rows) == 1
    assert rows[0]["code"] == "45378"


def test_procedures_filtered_by_text(synthetic_db):
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = procedures_sql(
        source_id="src-john-1962",
        procedure_text="appendix",
    )
    rows = adapter.fetch_all(sql, params)
    assert len(rows) >= 2
