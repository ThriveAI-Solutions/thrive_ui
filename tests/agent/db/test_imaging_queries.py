from agent.db.analytics_adapter import AnalyticsDbAdapter
from agent.db.queries.imaging import imaging_sql


def test_imaging_unions_orders_and_documents(synthetic_db):
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = imaging_sql(source_id="src-john-1962")
    rows = adapter.fetch_all(sql, params)
    sources = {r["source"] for r in rows}
    assert "orders" in sources
    assert "documents" in sources


def test_imaging_filtered_by_modality_xray(synthetic_db):
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = imaging_sql(source_id="src-john-1962", modality="xray")
    rows = adapter.fetch_all(sql, params)
    assert len(rows) >= 1
    assert all("ray" in (r["description"] or "").lower() or "ray" in (r["mnemonic"] or "").lower() for r in rows)
