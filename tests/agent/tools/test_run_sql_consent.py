"""run_sql consent gate (#244).

Under enforcement, freeform SQL against patient-bearing views is refused so it
can't bypass the consent-gated curated tools. Non-patient SQL and the
enforcement-off path are unaffected.
"""

from unittest.mock import MagicMock

import pytest
from pydantic_ai.exceptions import ModelRetry

from sqlalchemy import create_engine, text

from agent.consent.sql_gate import references_patient_view
from agent.db.analytics_adapter import AnalyticsDbAdapter
from agent.tools.run_sql import RunSqlInput, run_sql


@pytest.mark.parametrize(
    "sql, expected",
    [
        ("SELECT COUNT(*) FROM federated_demographic_v", True),
        ("SELECT * FROM internal_patient_profile_v", True),
        ("WITH x AS (SELECT 1) SELECT * FROM federated_results_v", True),
        ("SELECT * FROM metric_federated_data_v", True),
        ("SELECT 1", False),
        ("SELECT code, description FROM vocab_icd10", False),
    ],
)
def test_references_patient_view(sql, expected):
    assert references_patient_view(sql) is expected


def _ctx(*, enforce_consent):
    ctx = MagicMock()
    deps = MagicMock()
    deps.enforce_consent = enforce_consent
    deps.user_role = MagicMock(value=1)
    deps.sqlite_session = None
    deps.analytics_db = None  # not reached when the gate refuses first
    ctx.deps = deps
    return ctx


def test_enforcement_refuses_patient_sql():
    ctx = _ctx(enforce_consent=True)
    with pytest.raises(ModelRetry, match="Consent enforcement blocks"):
        run_sql(ctx, RunSqlInput(sql="SELECT source_id FROM federated_demographic_v"))


def test_enforcement_off_does_not_gate():
    # Enforcement off => the consent gate is skipped; execution proceeds to the
    # "analytics database not configured" retry (analytics_db is None here),
    # proving the consent gate did not fire.
    ctx = _ctx(enforce_consent=False)
    with pytest.raises(ModelRetry, match="Analytics database is not configured"):
        run_sql(ctx, RunSqlInput(sql="SELECT source_id FROM federated_demographic_v"))


# --- sqlglot 3-mode gate under enforcement (#244): aggregate queries pass but
# their measure cells are small-cell suppressed; row-level queries are refused;
# non-patient queries are untouched. ---


def _demographics_adapter():
    eng = create_engine("sqlite://")
    with eng.begin() as c:
        c.execute(text("CREATE TABLE federated_demographic_v (source_id TEXT, gender TEXT)"))
        for sid, g in [("s1", "F"), ("s2", "F"), ("s3", "M")]:
            c.execute(text("INSERT INTO federated_demographic_v VALUES (:s, :g)"), {"s": sid, "g": g})
    return AnalyticsDbAdapter(engine=eng, dialect="sqlite")


def _real_ctx(adapter, *, enforce_consent, threshold=11):
    ctx = MagicMock()
    deps = MagicMock()
    deps.enforce_consent = enforce_consent
    deps.user_role = MagicMock(value=1)
    deps.consent_bypass_roles = frozenset()
    deps.small_cell_threshold = threshold
    deps.analytics_db = adapter
    deps.sqlite_session = None
    ctx.deps = deps
    return ctx


def test_aggregate_count_suppressed_end_to_end():
    ctx = _real_ctx(_demographics_adapter(), enforce_consent=True)
    result = run_sql(ctx, RunSqlInput(sql="SELECT COUNT(*) AS n FROM federated_demographic_v"))
    # count of 3 is below the floor of 11 -> suppressed label, not the raw number.
    assert result.columns == ["n"]
    assert result.rows == [["fewer than 11"]]


def test_aggregate_group_labels_preserved_measures_suppressed():
    ctx = _real_ctx(_demographics_adapter(), enforce_consent=True)
    result = run_sql(
        ctx,
        RunSqlInput(sql="SELECT gender, COUNT(*) AS n FROM federated_demographic_v GROUP BY gender"),
    )
    by_gender = {row[0]: row[1] for row in result.rows}
    # Label column intact; every small count suppressed.
    assert set(by_gender) == {"F", "M"}
    assert all(v == "fewer than 11" for v in by_gender.values())


def test_aggregate_large_count_not_suppressed():
    ctx = _real_ctx(_demographics_adapter(), enforce_consent=True, threshold=2)
    result = run_sql(ctx, RunSqlInput(sql="SELECT COUNT(*) AS n FROM federated_demographic_v"))
    # 3 >= threshold 2 -> visible.
    assert result.rows == [[3]]


def test_row_level_patient_sql_refused():
    ctx = _real_ctx(_demographics_adapter(), enforce_consent=True)
    with pytest.raises(ModelRetry, match="row-level freeform SQL"):
        run_sql(ctx, RunSqlInput(sql="SELECT source_id FROM federated_demographic_v"))


def test_non_patient_sql_untouched_under_enforcement():
    ctx = _real_ctx(_demographics_adapter(), enforce_consent=True)
    result = run_sql(ctx, RunSqlInput(sql="SELECT 1 AS one"))
    assert result.rows == [[1]]
