from agent.db.analytics_adapter import AnalyticsDbAdapter
from agent.db.queries.documents import documents_sql


def test_documents_for_source_id(synthetic_db):
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = documents_sql(source_id="src-john-1962")
    rows = adapter.fetch_all(sql, params)
    assert len(rows) == 3


def test_documents_filtered_by_type(synthetic_db):
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = documents_sql(source_id="src-john-1962", document_type="progress")
    rows = adapter.fetch_all(sql, params)
    assert len(rows) == 1
    assert rows[0]["name"] == "Progress Note"


def test_documents_filtered_by_date(synthetic_db):
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = documents_sql(
        source_id="src-john-1962",
        start_date="2026-01-01",
    )
    rows = adapter.fetch_all(sql, params)
    assert len(rows) == 2
