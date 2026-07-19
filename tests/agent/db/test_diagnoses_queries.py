from agent.db.analytics_adapter import AnalyticsDbAdapter
from agent.db.queries.diagnoses import diagnoses_sql


def test_diagnoses_for_source_id(synthetic_db):
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = diagnoses_sql(source_id="src-john-1962")
    rows = adapter.fetch_all(sql, params)
    assert len(rows) == 4
    assert "status" in rows[0]


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


def test_diagnoses_order_by_puts_nulls_last():
    """Postgres/Redshift default NULLS FIRST on DESC, so without an explicit
    NULLS LAST a NULL-dated row wins most_recent_only's LIMIT 1 and shadows
    the real newest diagnosis."""
    sql, _ = diagnoses_sql(source_id="s", most_recent_only=True)
    assert "NULLS LAST" in sql


def test_diagnoses_projection_selects_only_expected_warehouse_fields():
    sql, _ = diagnoses_sql(source_id="s")
    projection = sql.split("SELECT", 1)[1].split("FROM", 1)[0]
    assert [column.strip() for column in projection.split(",")] == [
        "source_id",
        "code",
        "code_type",
        "diagnosis",
        "diagnosis_datetime",
        "status",
        "status_datetime",
        "chronic_ind",
        "service_provider_npi",
    ]


def test_diagnoses_combined_filters_and_order_keep_sql_shape():
    sql, params = diagnoses_sql(
        source_id="s",
        icd10_codes=["E11.9"],
        condition_text="Diabetes",
        most_recent_only=True,
    )
    normalized_sql = " ".join(sql.split())

    assert "WHERE source_id = :source_id AND code_type IN (" in normalized_sql
    assert "AND code IN (:dc_0) AND LOWER(diagnosis) LIKE :ct" in normalized_sql
    assert normalized_sql.endswith("ORDER BY diagnosis_datetime DESC NULLS LAST LIMIT 1")
    assert params["source_id"] == "s"
    assert params["dc_0"] == "E11.9"
    assert params["ct"] == "%diabetes%"
