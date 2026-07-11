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
        run_logger=MagicMock(),
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


def test_run_sql_wraps_db_errors_as_model_retry(synthetic_db):
    """Live regression on 2026-05-13: gpt-oss called run_sql with
    `SELECT p.source_id FROM dw.internal_patient_profile_v p ...` —
    the column doesn't exist on that view, psycopg2 raised
    UndefinedColumn, and the unwrapped SQLAlchemyError killed the whole
    agent stream. The tool must convert DB errors to ModelRetry so the
    LLM can read the underlying message and fix its query on the next
    turn instead of aborting the run."""
    from agent.tools.run_sql import run_sql, RunSqlInput

    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, _selected_john())
    # Reference a column that doesn't exist on synthetic_db's schema —
    # SQLite reports OperationalError which sqlalchemy wraps in
    # SQLAlchemyError. The tool should catch and re-raise as ModelRetry.
    with pytest.raises(ModelRetry) as excinfo:
        run_sql(ctx, RunSqlInput(sql="SELECT no_such_column FROM federated_demographic_v"))
    msg = str(excinfo.value)
    assert "SQL execution failed" in msg
    # Underlying DB error should be surfaced verbatim so the model can
    # actually act on it.
    assert "no_such_column" in msg.lower() or "no such column" in msg.lower()


# {{codes:<set_id>}} macro tests -----------------------------------------
# expand_code_macros itself is pure-mock tested in test_sql_macros.py; these
# tests verify the tool wires it in correctly: expansion happens BEFORE the
# AST guard, the executed SQL is the expanded form, result.sql echoes the
# original macro-form SQL, and unknown sets / macro-free SQL are handled
# per the security-sensitive invariants in the Task 4 brief.


def test_run_sql_macro_expands_before_guard_and_executes(synthetic_db, vocab_session):
    from agent.tools.run_sql import run_sql, RunSqlInput

    ctx = MagicMock()
    deps = _deps(synthetic_db, _selected_john())
    deps.sqlite_session = vocab_session
    ctx.deps = deps

    result = run_sql(
        ctx,
        RunSqlInput(sql="SELECT 'E11.9' AS code WHERE 'E11.9' IN {{codes:dx:diabetes-mellitus}}"),
    )

    # The executed (expanded) SQL is recorded for provenance and contains
    # the quoted dotted + undotted match-forms fetched from the vocab DB.
    assert "'E11.9'" in ctx.deps.last_sql
    assert "'E119'" in ctx.deps.last_sql
    assert "{{codes:" not in ctx.deps.last_sql
    # result.sql echoes the ORIGINAL macro-form SQL (chiron parity).
    assert "{{codes:dx:diabetes-mellitus}}" in result.sql
    assert result.row_count == 1
    assert result.rows[0] == ["E11.9"]


def test_run_sql_macro_unknown_set_raises_actionable_model_retry(synthetic_db, vocab_session):
    from agent.tools.run_sql import run_sql, RunSqlInput

    ctx = MagicMock()
    deps = _deps(synthetic_db, _selected_john())
    deps.sqlite_session = vocab_session
    ctx.deps = deps

    with pytest.raises(ModelRetry) as excinfo:
        run_sql(ctx, RunSqlInput(sql="SELECT 1 WHERE 1 IN {{codes:dx:not-a-real-set}}"))
    msg = str(excinfo.value)
    assert "not-a-real-set" in msg
    assert "search_codes first" in msg
    assert "{{codes:<set_id>}}" in msg


def test_run_sql_macro_reliability_note_combines_with_truncation_note(synthetic_db, vocab_session):
    """reliability_note carries the macro-expansion note AND the truncation
    note, joined with '; ', per the brief's combining-logic requirement."""
    from sqlalchemy import text
    from agent.tools.run_sql import run_sql, RunSqlInput

    with synthetic_db.connect() as conn:
        conn.execute(text("CREATE TABLE macro_big_table (a INTEGER)"))
        for i in range(600):
            conn.execute(text("INSERT INTO macro_big_table (a) VALUES (:a)"), {"a": i})
        conn.commit()

    ctx = MagicMock()
    deps = _deps(synthetic_db, _selected_john())
    deps.sqlite_session = vocab_session
    ctx.deps = deps

    result = run_sql(
        ctx,
        RunSqlInput(sql="SELECT a FROM macro_big_table WHERE 'E11.9' IN {{codes:dx:diabetes-mellitus}} OR 1=1"),
    )
    assert result.truncated is True
    assert result.reliability_note is not None
    assert "expanded to" in result.reliability_note
    assert "code match-forms" in result.reliability_note
    assert "Results truncated at 500 rows" in result.reliability_note
    assert "; " in result.reliability_note


def test_run_sql_macro_free_sql_makes_zero_db_calls(synthetic_db):
    """Macro-free SQL must not touch the vocab session at all."""
    from agent.tools.run_sql import run_sql, RunSqlInput

    session = MagicMock(name="sqlite_session")
    ctx = MagicMock()
    deps = _deps(synthetic_db, _selected_john())
    deps.sqlite_session = session
    ctx.deps = deps

    result = run_sql(ctx, RunSqlInput(sql="SELECT 1 AS a"))

    assert result.row_count == 1
    session.assert_not_called()
    session.get.assert_not_called()
    session.scalars.assert_not_called()
