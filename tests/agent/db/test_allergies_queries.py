from agent.db.analytics_adapter import AnalyticsDbAdapter
from agent.db.queries.allergies import allergies_sql


def test_allergies_default_returns_active_only(synthetic_db):
    """Per epic #201 AC: default behavior returns active allergies; resolved
    allergies are filtered out unless include_inactive=True."""
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = allergies_sql(source_id="src-john-1962")
    rows = adapter.fetch_all(sql, params)
    assert len(rows) == 2
    assert {r["allergy"] for r in rows} == {"Penicillin", "Peanuts"}


def test_allergies_include_inactive_returns_all(synthetic_db):
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = allergies_sql(source_id="src-john-1962", include_inactive=True)
    rows = adapter.fetch_all(sql, params)
    assert len(rows) == 3
    assert {r["allergy"] for r in rows} == {"Penicillin", "Peanuts", "Latex"}


def test_allergies_filtered_by_category(synthetic_db):
    """category='drug' maps to type IN ('Drug allergy', ...)."""
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = allergies_sql(source_id="src-john-1962", category="drug")
    rows = adapter.fetch_all(sql, params)
    assert len(rows) == 1
    assert rows[0]["allergy"] == "Penicillin"


def test_allergies_filtered_by_text(synthetic_db):
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = allergies_sql(source_id="src-john-1962", allergen_text="peanut")
    rows = adapter.fetch_all(sql, params)
    assert len(rows) == 1
    assert rows[0]["allergy"] == "Peanuts"


def test_allergies_filtered_by_snomed_codes(synthetic_db):
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = allergies_sql(source_id="src-john-1962", snomed_codes=["91936005"])
    rows = adapter.fetch_all(sql, params)
    assert len(rows) == 1
    assert rows[0]["allergy"] == "Penicillin"


def test_allergies_no_records_for_unknown_patient(synthetic_db):
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = allergies_sql(source_id="src-jane-1985")
    rows = adapter.fetch_all(sql, params)
    assert rows == []


def test_allergies_nka_row_is_returned_verbatim(synthetic_db):
    """'NO KNOWN ALLERGIES' rows surface at the SQL layer the same as any
    other row. The tool-envelope layer is responsible for interpreting them
    as a first-class negative assertion."""
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = allergies_sql(source_id="src-john-1971")
    rows = adapter.fetch_all(sql, params)
    assert len(rows) == 1
    assert rows[0]["allergy"] == "NO KNOWN ALLERGIES"


def test_allergies_filtered_by_date_range(synthetic_db):
    """date_range filters by COALESCE(date, created_date). The clinical `date`
    is frequently NULL in production, so rows fall back to created_date rather
    than being silently dropped by the filter."""
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = allergies_sql(
        source_id="src-john-1962",
        include_inactive=True,
        start_date="2018-01-01",
    )
    rows = adapter.fetch_all(sql, params)
    # Peanuts (date 2018-03-01) kept; Latex (date NULL → created_date 2020-06-01)
    # kept via the COALESCE fallback; Penicillin (date 2010-05-01) filtered out.
    assert {r["allergy"] for r in rows} == {"Peanuts", "Latex"}


def test_allergies_event_datetime_falls_back_to_created_date(synthetic_db):
    """When the clinical `date` is NULL, event_datetime must surface
    created_date rather than NULL so the row still sorts and date-filters."""
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = allergies_sql(source_id="src-john-1962", include_inactive=True)
    rows = adapter.fetch_all(sql, params)
    latex = next(r for r in rows if r["allergy"] == "Latex")
    assert latex["onset_date"] is None  # clinical date genuinely unknown
    assert str(latex["event_datetime"]).startswith("2020-06-01")


def test_allergies_schema_prefix_applied(synthetic_db):
    """The SQL builder must accept schema_prefix and qualify the view."""
    sql, _ = allergies_sql(source_id="src-john-1962", schema_prefix="dw.")
    assert "dw.federated_allergies_v" in sql
