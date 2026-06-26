import pytest

from agent.db.queries.adt import (
    inpatient_admission_flag_sql,
    inpatient_cohort_subquery_sql,
    qualifying_admit_date_sql,
    PREADMIT_PENDING_STATUSES,
)


def test_flag_sql_uses_bool_or_for_redshift():
    sql = inpatient_admission_flag_sql("redshift", alias="adt")
    assert "BOOL_OR(" in sql
    assert "NOT BOOL_OR(adt.clean_status = 'CANCEL ADMIT')" in sql
    # evidence: inpatient setting OR A06 conversion
    assert "UPPER(adt.clean_setting) IN ('INPATIENT')" in sql
    assert "adt.clean_status = 'A06'" in sql
    # NULL-safe exclusions
    assert "COALESCE(adt.clean_status, '') NOT IN ('A05', 'A14', 'A38', 'A27')" in sql
    assert "COALESCE(adt.cancelled_flag, 'N') <> 'Y'" in sql


def test_flag_sql_uses_case_max_for_sqlite():
    sql = inpatient_admission_flag_sql("sqlite", alias="adt")
    assert "BOOL_OR" not in sql
    assert "MAX(CASE WHEN" in sql
    assert "= 1" in sql and "= 0" in sql


def test_flag_sql_rejects_unknown_dialect():
    with pytest.raises(ValueError):
        inpatient_admission_flag_sql("oracle")


def test_qualifying_admit_date_sql_is_conditional_min():
    sql = qualifying_admit_date_sql(alias="adt")
    assert sql.startswith("MIN(CASE WHEN")
    assert "adt.event_date" in sql


def test_preadmit_pending_constant_values():
    assert PREADMIT_PENDING_STATUSES == ("A05", "A14", "A38", "A27")


def test_cohort_subquery_distinct_for_filter():
    sql, params = inpatient_cohort_subquery_sql("redshift", schema_prefix="dw.")
    n = " ".join(sql.split())
    assert "SELECT DISTINCT patient_id" in n
    assert "FROM dw.federated_adt_v adt" in n
    assert "GROUP BY adt.patient_id, adt.visit_number" in n
    assert "HAVING BOOL_OR(" in n
    assert "qualifying_admit_date" not in n
    assert params == {}


def test_cohort_subquery_projects_admit_date_for_breakdown():
    sql, params = inpatient_cohort_subquery_sql("sqlite", project_admit_date=True)
    n = " ".join(sql.split())
    assert "SELECT adt.patient_id" in n and "DISTINCT" not in n
    assert "AS qualifying_admit_date" in n


def test_cohort_subquery_date_params():
    sql, params = inpatient_cohort_subquery_sql(
        "redshift", start_date="2026-01-01", end_date="2026-12-31", param_prefix="adt"
    )
    assert params == {"adt_start": "2026-01-01", "adt_end": "2026-12-31"}
    assert ":adt_start" in sql and ":adt_end" in sql
