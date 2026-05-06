from agent.db.analytics_adapter import AnalyticsDbAdapter
from agent.db.queries.diagnoses import diagnoses_sql


def test_diagnoses_for_source_id(synthetic_db):
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = diagnoses_sql(source_id="src-john-1962")
    rows = adapter.fetch_all(sql, params)
    assert len(rows) == 3


def test_diagnoses_filtered_by_icd10_codes(synthetic_db):
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = diagnoses_sql(
        source_id="src-john-1962",
        icd10_codes=["E11.9"],
    )
    rows = adapter.fetch_all(sql, params)
    assert len(rows) == 1
    assert rows[0]["diagnosis"].startswith("Type 2 diabetes")


def test_diagnoses_filtered_by_text(synthetic_db):
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = diagnoses_sql(
        source_id="src-john-1962",
        condition_text="diabetes",
    )
    rows = adapter.fetch_all(sql, params)
    assert len(rows) == 1


def test_diagnoses_most_recent_only(synthetic_db):
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = diagnoses_sql(
        source_id="src-john-1962",
        most_recent_only=True,
    )
    rows = adapter.fetch_all(sql, params)
    assert len(rows) == 1
    assert rows[0]["code"] == "B16.9"
