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


def test_procedures_excludes_non_pcs_icd10_diagnoses(synthetic_db):
    """ICD-10 diagnosis codes (E11.9 diabetes, B16.9 hep B) must NOT
    appear in the procedures UNION — only ICD-10-PCS rows belong here."""
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = procedures_sql(source_id="src-john-1962")
    rows = adapter.fetch_all(sql, params)
    problem_rows = [r for r in rows if r["source"] == "problems"]
    assert all(r["code_type"] == "ICD-10-PCS" for r in problem_rows), (
        f"Non-PCS rows leaked into procedures: {[r for r in problem_rows if r['code_type'] != 'ICD-10-PCS']}"
    )
    # The diabetes diagnosis must not be in the result at all.
    assert not any(r["code"] == "E11.9" for r in rows)
