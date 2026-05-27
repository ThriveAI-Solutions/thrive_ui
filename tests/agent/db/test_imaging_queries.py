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


def test_imaging_body_region_chest_uses_keywords(synthetic_db):
    """body_region='chest' should match Chest X-ray order via keyword expansion."""
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = imaging_sql(source_id="src-john-1962", body_region="chest")
    rows = adapter.fetch_all(sql, params)
    assert len(rows) >= 1
    # The Chest X-ray order should match
    descriptions = [r["description"].lower() for r in rows if r["description"]]
    assert any("chest" in d for d in descriptions)


def test_imaging_body_region_knee_sql_is_valid(synthetic_db):
    """body_region='knee' generates valid SQL; no knee imaging row in test data so result may be empty."""
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = imaging_sql(source_id="src-john-1962", body_region="knee")
    rows = adapter.fetch_all(sql, params)
    # knee keyword expansion produces valid SQL; result depends on fixture data
    assert isinstance(rows, list)


def test_imaging_body_region_fallback_for_unknown_region(synthetic_db):
    """Unknown body region should fall back to raw LIKE matching."""
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = imaging_sql(source_id="src-john-1962", body_region="elbow")
    rows = adapter.fetch_all(sql, params)
    # No imaging rows should match 'elbow' in our test data
    assert isinstance(rows, list)
