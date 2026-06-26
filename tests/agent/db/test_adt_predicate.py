import pytest

from agent.db.queries.adt import (
    inpatient_admission_flag_sql,
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
