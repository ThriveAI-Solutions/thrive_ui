"""run_sql tool tests.

Per Phase 3 design §3.4 — no role gating, no table whitelist; defenses
are AST guard, read-only engine, 500-row cap, 30s timeout.
"""

from __future__ import annotations
from datetime import date, datetime
from unittest.mock import MagicMock
import pytest
from pydantic import ValidationError
from pydantic_ai import ModelRetry

from agent.deps import AgentDeps, SelectedPatient
from agent.db.analytics_adapter import AnalyticsDbAdapter


def _selected_john() -> SelectedPatient:
    return SelectedPatient(
        source_id="src-john-1962",
        display_name="John Smith",
        dob=date(1962, 5, 1),
        selected_at=datetime.now(),
        selection_origin="user_click",
    )


def _deps(synthetic_db, selected: SelectedPatient | None) -> AgentDeps:
    return AgentDeps(
        user_id=1,
        user_role=MagicMock(value=1),
        session_id="s1",
        selected_patient=selected,
        last_dataframe=None,
        last_sql=None,
        last_query_meta=None,
        analytics_db=AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite"),
        rag=None,
        sqlite_session=None,
        audit_logger=MagicMock(),
    )


def test_run_sql_select_returns_result_and_sets_last_dataframe(synthetic_db):
    from agent.tools.run_sql import run_sql, RunSqlInput
    import pandas as pd

    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, _selected_john())
    result = run_sql(ctx, RunSqlInput(sql="SELECT 1 AS a, 'x' AS b"))

    assert result.row_count == 1
    assert result.columns == ["a", "b"]
    assert result.rows[0] == [1, "x"]
    assert result.truncated is False
    assert isinstance(ctx.deps.last_dataframe, pd.DataFrame)
    assert ctx.deps.last_dataframe["a"].tolist() == [1]


def test_run_sql_rejects_ddl(synthetic_db):
    from agent.tools.run_sql import run_sql, RunSqlInput

    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, _selected_john())
    with pytest.raises(ModelRetry, match="read-only|write keyword"):
        run_sql(ctx, RunSqlInput(sql="DROP TABLE t"))


def test_run_sql_rejects_dml(synthetic_db):
    from agent.tools.run_sql import run_sql, RunSqlInput

    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, _selected_john())
    with pytest.raises(ModelRetry, match="read-only|write keyword"):
        run_sql(ctx, RunSqlInput(sql="DELETE FROM t"))


def test_run_sql_rejects_multiple_statements(synthetic_db):
    from agent.tools.run_sql import run_sql, RunSqlInput

    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, _selected_john())
    with pytest.raises(ModelRetry, match="single statement"):
        run_sql(ctx, RunSqlInput(sql="SELECT 1; SELECT 2"))


def test_run_sql_rejects_pg_catalog(synthetic_db):
    from agent.tools.run_sql import run_sql, RunSqlInput

    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, _selected_john())
    with pytest.raises(ModelRetry, match="system table"):
        run_sql(ctx, RunSqlInput(sql="SELECT * FROM pg_catalog.pg_user"))


def test_run_sql_truncates_at_500_rows(synthetic_db):
    """Build a 600-row table on the fly and confirm row cap kicks in."""
    from sqlalchemy import text
    from agent.tools.run_sql import run_sql, RunSqlInput

    with synthetic_db.connect() as conn:
        conn.execute(text("CREATE TABLE big_table (a INTEGER)"))
        for i in range(600):
            conn.execute(text("INSERT INTO big_table (a) VALUES (:a)"), {"a": i})
        conn.commit()

    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, _selected_john())
    result = run_sql(ctx, RunSqlInput(sql="SELECT a FROM big_table"))

    assert result.row_count == 500
    assert result.truncated is True


def test_run_sql_with_existing_smaller_limit_is_respected(synthetic_db):
    from agent.tools.run_sql import run_sql, RunSqlInput

    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, _selected_john())
    result = run_sql(
        ctx,
        RunSqlInput(sql="SELECT 1 AS a UNION SELECT 2 UNION SELECT 3 LIMIT 2"),
    )
    assert result.row_count <= 2
    assert result.truncated is False


def test_run_sql_input_validation_rejects_empty():
    from agent.tools.run_sql import RunSqlInput

    with pytest.raises(ValidationError):
        RunSqlInput(sql="   ")
