from agent.db.analytics_adapter import AnalyticsDbAdapter
from agent.db.queries.labs import labs_sql


def test_labs_for_source_id(synthetic_db):
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = labs_sql(source_id="src-john-1962")
    rows = adapter.fetch_all(sql, params)
    assert len(rows) == 4


def test_labs_filtered_by_loinc_codes(synthetic_db):
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = labs_sql(
        source_id="src-john-1962",
        loinc_codes=["4548-4"],
    )
    rows = adapter.fetch_all(sql, params)
    assert len(rows) == 1
    assert rows[0]["name"] == "Hemoglobin A1c"


def test_labs_filtered_by_date_range(synthetic_db):
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = labs_sql(
        source_id="src-john-1962",
        start_date="2026-02-01",
        end_date="2026-03-31",
    )
    rows = adapter.fetch_all(sql, params)
    assert len(rows) == 2


def test_labs_filtered_by_test_name_text(synthetic_db):
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = labs_sql(
        source_id="src-john-1962",
        test_name_text="hep",
    )
    rows = adapter.fetch_all(sql, params)
    assert len(rows) == 1
    assert "HBsAg" in rows[0]["name"]


def test_labs_negative_result_filter(synthetic_db):
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = labs_sql(
        source_id="src-john-1962",
        result_filter="negative",
    )
    rows = adapter.fetch_all(sql, params)
    assert len(rows) == 1
    assert rows[0]["clean_result"] == "negative"


def test_labs_loinc_filter_uses_code_type_variants(synthetic_db):
    """If a fixture row has code_type 'loinc' (lowercase) or 'LN', it still matches."""
    from agent.code_normalizer import variants_for

    assert "LOINC" in variants_for("loinc")
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = labs_sql(
        source_id="src-john-1962",
        loinc_codes=["LOC-CHEM-99"],
    )
    rows = adapter.fetch_all(sql, params)
    assert rows == []
