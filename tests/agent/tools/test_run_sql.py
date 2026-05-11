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


def test_run_sql_rejects_cte_wrapped_dml(synthetic_db):
    """The first-keyword check passes (WITH) but the body contains DELETE,
    which the token-level scan must catch."""
    from agent.tools.run_sql import run_sql, RunSqlInput

    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, _selected_john())
    with pytest.raises(ModelRetry, match="write keyword"):
        run_sql(
            ctx,
            RunSqlInput(sql="WITH x AS (SELECT 1 AS n) DELETE FROM federated_problems_v"),
        )


def test_run_sql_rejects_cte_wrapped_update(synthetic_db):
    from agent.tools.run_sql import run_sql, RunSqlInput

    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, _selected_john())
    with pytest.raises(ModelRetry, match="write keyword"):
        run_sql(
            ctx,
            RunSqlInput(sql="WITH x AS (UPDATE t SET a=1 RETURNING *) SELECT * FROM x"),
        )


def test_run_sql_block_comment_drop_caught_by_defense_in_depth(synthetic_db):
    """Spec §6 risk #2: even though the AST guard treats DROP inside a
    /* */ comment as benign (it's not a keyword token), the adapter-level
    regex catches the literal `DROP` word as a belt-and-suspenders backstop.

    Net effect: bypass attempts via comment-encoded DDL are rejected. The
    user pays a false-positive cost on the harmless `/* DROP TABLE */
    SELECT 1` pattern — acceptable, since the same query without the
    misleading comment works fine."""
    from agent.tools.run_sql import run_sql, RunSqlInput

    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, _selected_john())
    with pytest.raises(ModelRetry, match="read-only|write keyword"):
        run_sql(ctx, RunSqlInput(sql="/* DROP TABLE foo */ SELECT 1 AS a"))


def test_run_sql_trailing_line_comment_does_not_bypass_row_cap(synthetic_db):
    """Regression for trailing -- comment defeating LIMIT injection.

    Without the comment-stripping in _inject_limit, the appended
    ` LIMIT 501` lands inside the comment and is ignored, leaving the
    query effectively uncapped."""
    from sqlalchemy import text
    from agent.tools.run_sql import run_sql, RunSqlInput

    with synthetic_db.connect() as conn:
        conn.execute(text("CREATE TABLE big_table_cmt (a INTEGER)"))
        for i in range(600):
            conn.execute(text("INSERT INTO big_table_cmt (a) VALUES (:a)"), {"a": i})
        conn.commit()

    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, _selected_john())
    result = run_sql(
        ctx,
        RunSqlInput(sql="SELECT a FROM big_table_cmt -- give me everything"),
    )
    assert result.row_count == 500
    assert result.truncated is True


def test_run_sql_trailing_block_comment_does_not_bypass_row_cap(synthetic_db):
    from sqlalchemy import text
    from agent.tools.run_sql import run_sql, RunSqlInput

    with synthetic_db.connect() as conn:
        conn.execute(text("CREATE TABLE big_table_blk (a INTEGER)"))
        for i in range(600):
            conn.execute(text("INSERT INTO big_table_blk (a) VALUES (:a)"), {"a": i})
        conn.commit()

    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, _selected_john())
    result = run_sql(
        ctx,
        RunSqlInput(sql="SELECT a FROM big_table_blk /* and please /* nested */ everything */"),
    )
    assert result.row_count == 500
    assert result.truncated is True


def test_run_sql_rejects_merge_vacuum_analyze_copy(synthetic_db):
    """Phase 3 closeout review #5: MERGE/VACUUM/ANALYZE/COPY must be
    rejected at both the tool AST guard and the adapter regex guard."""
    from agent.tools.run_sql import run_sql, RunSqlInput

    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, _selected_john())
    for stmt in (
        "MERGE INTO t USING s ON t.id = s.id WHEN MATCHED THEN DELETE",
        "VACUUM",
        "ANALYZE t",
        "COPY t FROM '/tmp/x.csv'",
    ):
        with pytest.raises(ModelRetry):
            run_sql(ctx, RunSqlInput(sql=stmt))
