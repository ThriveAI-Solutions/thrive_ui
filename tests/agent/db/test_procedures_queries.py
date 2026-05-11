from sqlalchemy import text

from agent.db.analytics_adapter import AnalyticsDbAdapter
from agent.db.queries.procedures import procedures_sql


def test_procedures_unions_orders_and_problems(synthetic_db):
    """The claims branch is hard-suppressed in Phase 3 (see procedures.py for
    rationale); the UNION resolves to orders + problems only."""
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = procedures_sql(source_id="src-john-1962")
    rows = adapter.fetch_all(sql, params)
    sources = {r["source"] for r in rows}
    assert sources == {"orders", "problems"}


def test_procedures_claims_branch_is_suppressed_for_other_patients(synthetic_db):
    """HIPAA guardrail: federated_claims_icd_procedure_detail_v has no
    patient identifier. A naive query would leak every patient's claims
    rows. Confirm that even with a 'wrong patient' claims row in the
    fixture, the claims branch yields zero rows for any source_id."""
    with synthetic_db.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO federated_claims_icd_procedure_detail_v VALUES "
                "('CLM-OTHER-99', '0DTJ4ZZ', 'ICD-10-PCS', 1, '2024-09-01', "
                "1, '2024-09-01', 'claims_other.csv', 'ICD', 'Highmark')"
            )
        )

    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    # Both a valid patient's source_id and a bogus one must return zero
    # claims rows; the suppression doesn't depend on whether source_id
    # actually exists in any source table.
    for src in ("src-john-1962", "src-does-not-exist"):
        sql, params = procedures_sql(source_id=src)
        rows = adapter.fetch_all(sql, params)
        claims_rows = [r for r in rows if r["source"] == "claims"]
        assert claims_rows == [], f"Claims branch leaked rows for source_id={src!r}: {claims_rows}"


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
    """The text filter applies to the orders and problems branches only —
    federated_claims_icd_procedure_detail_v has no description column to
    match against, and the branch is suppressed anyway."""
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    sql, params = procedures_sql(
        source_id="src-john-1962",
        procedure_text="appendix",
    )
    rows = adapter.fetch_all(sql, params)
    descriptions = [r["description"] for r in rows if r.get("description")]
    assert len(rows) >= 1
    assert all("appendix" in d.lower() for d in descriptions)


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
