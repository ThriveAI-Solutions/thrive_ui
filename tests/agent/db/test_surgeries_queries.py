from agent.db.analytics_adapter import AnalyticsDbAdapter
from agent.db.queries.surgeries import surgeries_sql


def test_surgeries_includes_surgical_cpt(synthetic_db):
    """CPT 27447 (knee arthroplasty) is in surgery range 10004-69990."""
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = surgeries_sql(source_id="src-john-1962")
    rows = adapter.fetch_all(sql, params)
    codes = {r["code"] for r in rows if r["source"] == "orders"}
    assert "27447" in codes


def test_surgeries_includes_colonoscopy_in_surgery_range(synthetic_db):
    """CPT 45378 (colonoscopy) is in 40490-49999 digestive surgery range."""
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = surgeries_sql(source_id="src-john-1962")
    rows = adapter.fetch_all(sql, params)
    codes = {r["code"] for r in rows if r["source"] == "orders"}
    assert "45378" in codes


def test_surgeries_excludes_non_surgical_cpt(synthetic_db):
    """CPT 71046 (chest x-ray) is in 70000-79999 radiology — outside surgery range."""
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = surgeries_sql(source_id="src-john-1962")
    rows = adapter.fetch_all(sql, params)
    codes = {r["code"] for r in rows if r["source"] == "orders"}
    assert "71046" not in codes


def test_surgeries_excludes_non_cpt_orders(synthetic_db):
    """LOC-X-1 has empty code_type — should be excluded by CPT filter."""
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = surgeries_sql(source_id="src-john-1962")
    rows = adapter.fetch_all(sql, params)
    codes = {r["code"] for r in rows if r["source"] == "orders"}
    assert "LOC-X-1" not in codes


def test_surgeries_includes_invasive_pcs(synthetic_db):
    """0DTJ4ZZ has root op T (Resection) — invasive."""
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = surgeries_sql(source_id="src-john-1962")
    rows = adapter.fetch_all(sql, params)
    problem_codes = {r["code"] for r in rows if r["source"] == "problems"}
    assert "0DTJ4ZZ" in problem_codes


def test_surgeries_excludes_inspection_pcs(synthetic_db):
    """0WJG4ZZ has root op J (Inspection) — non-invasive, excluded."""
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = surgeries_sql(source_id="src-john-1962")
    rows = adapter.fetch_all(sql, params)
    problem_codes = {r["code"] for r in rows if r["source"] == "problems"}
    assert "0WJG4ZZ" not in problem_codes


def test_surgeries_claims_branch_suppressed(synthetic_db):
    """Claims branch is suppressed (AND 1=0) because the table has no patient identifier."""
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = surgeries_sql(source_id="src-john-1962")
    rows = adapter.fetch_all(sql, params)
    claims_rows = [r for r in rows if r["source"] == "claims"]
    assert len(claims_rows) == 0


def test_surgeries_performing_provider_lists_all_when_ambiguous(synthetic_db):
    """The knee arthroplasty on 2025-06-15 matches two encounters → comma-joined list."""
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = surgeries_sql(source_id="src-john-1962")
    rows = adapter.fetch_all(sql, params)
    knee = [r for r in rows if r["code"] == "27447"]
    assert len(knee) == 1
    provider = knee[0]["performing_provider"]
    assert "Dr. Ortho" in provider
    assert "Dr. Anesthesia" in provider


def test_surgeries_performing_provider_null_when_no_encounter(synthetic_db):
    """The appendix resection on 2024-08-22 has no matching encounter → NULL provider."""
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = surgeries_sql(source_id="src-john-1962")
    rows = adapter.fetch_all(sql, params)
    appendix_orders = [r for r in rows if r["code"] == "0DTJ4ZZ" and r["source"] == "problems"]
    assert len(appendix_orders) == 1
    assert appendix_orders[0]["performing_provider"] is None


def test_surgeries_date_range_filter(synthetic_db):
    """Date range 2025-01-01 to 2025-12-31 should include knee arthroplasty but exclude appendix resection (2024)."""
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = surgeries_sql(
        source_id="src-john-1962",
        start_date="2025-01-01",
        end_date="2025-12-31",
    )
    rows = adapter.fetch_all(sql, params)
    codes = {r["code"] for r in rows}
    assert "27447" in codes
    assert "0DTJ4ZZ" not in codes


def test_surgeries_text_filter(synthetic_db):
    """Text filter 'knee' should match the arthroplasty row."""
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = surgeries_sql(
        source_id="src-john-1962",
        procedure_text="knee",
    )
    rows = adapter.fetch_all(sql, params)
    assert len(rows) == 1
    assert rows[0]["code"] == "27447"


def test_surgeries_cpt_code_filter(synthetic_db):
    """Explicit CPT code filter narrows to just that code."""
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = surgeries_sql(
        source_id="src-john-1962",
        cpt_codes=["27447"],
    )
    rows = adapter.fetch_all(sql, params)
    order_rows = [r for r in rows if r["source"] == "orders"]
    assert len(order_rows) == 1
    assert order_rows[0]["code"] == "27447"
