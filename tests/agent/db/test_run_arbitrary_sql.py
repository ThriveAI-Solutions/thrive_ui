"""run_arbitrary_sql is the adapter-level entry point for the run_sql
tool. It enforces the row cap and statement timeout that the tool's
AST guard cannot enforce.

Three engine behaviors here:
  * sqlite: query_only PRAGMA + LIMIT injection (timeout is a no-op in SQLite)
  * postgres: SESSION READ ONLY + LIMIT injection + statement_timeout
  * redshift: SESSION READ ONLY + LIMIT injection + statement_timeout

Tests run against an in-memory SQLite engine; integration with real
Redshift is covered by the manual dev-server smoke in Task 17.
"""

from __future__ import annotations
import pytest
from sqlalchemy import create_engine, text

from agent.db.analytics_adapter import AnalyticsDbAdapter


@pytest.fixture
def sqlite_adapter():
    engine = create_engine("sqlite:///:memory:")
    with engine.connect() as conn:
        conn.execute(text("CREATE TABLE t (a INTEGER, b TEXT)"))
        for i in range(10):
            conn.execute(
                text("INSERT INTO t (a, b) VALUES (:a, :b)"),
                {"a": i, "b": f"row{i}"},
            )
        conn.commit()
    return AnalyticsDbAdapter(engine=engine, dialect="sqlite", schema="")


def test_run_arbitrary_sql_returns_columns_and_rows(sqlite_adapter):
    columns, rows, truncated = sqlite_adapter.run_arbitrary_sql(
        "SELECT a, b FROM t ORDER BY a", row_cap=500, timeout_s=30
    )
    assert columns == ["a", "b"]
    assert len(rows) == 10
    assert rows[0] == [0, "row0"]
    assert truncated is False


def test_run_arbitrary_sql_truncates_at_row_cap(sqlite_adapter):
    columns, rows, truncated = sqlite_adapter.run_arbitrary_sql("SELECT a FROM t ORDER BY a", row_cap=3, timeout_s=30)
    assert len(rows) == 3
    assert truncated is True


def test_run_arbitrary_sql_existing_limit_kept_when_smaller(sqlite_adapter):
    """If the user wrote LIMIT 2, respect it — don't escalate to 500."""
    columns, rows, truncated = sqlite_adapter.run_arbitrary_sql(
        "SELECT a FROM t ORDER BY a LIMIT 2", row_cap=500, timeout_s=30
    )
    assert len(rows) == 2
    assert truncated is False


def test_run_arbitrary_sql_rejects_write_keywords(sqlite_adapter):
    """Belt-and-suspenders. The tool-level AST guard catches most writes
    but this is the last line of defense before the engine."""
    with pytest.raises(ValueError, match="read-only"):
        sqlite_adapter.run_arbitrary_sql("DELETE FROM t WHERE a > 5", row_cap=500, timeout_s=30)


def test_run_arbitrary_sql_writes_to_sql_log(sqlite_adapter):
    """For consistency with fetch_all, run_arbitrary_sql also logs
    so the tool-call card shows what SQL ran."""
    sqlite_adapter.run_arbitrary_sql("SELECT a FROM t ORDER BY a", row_cap=500, timeout_s=30)
    log = sqlite_adapter.pop_sql_log()
    assert len(log) == 1
    assert "SELECT a FROM t" in log[0]["sql"]


def test_run_arbitrary_sql_caps_query_ending_in_line_comment(sqlite_adapter):
    """Regression: a trailing -- comment used to let LIMIT injection land
    inside the comment, leaving the query un-capped."""
    columns, rows, truncated = sqlite_adapter.run_arbitrary_sql(
        "SELECT a FROM t ORDER BY a -- everything please",
        row_cap=3,
        timeout_s=30,
    )
    assert len(rows) == 3
    assert truncated is True


def test_run_arbitrary_sql_caps_query_ending_in_block_comment(sqlite_adapter):
    columns, rows, truncated = sqlite_adapter.run_arbitrary_sql(
        "SELECT a FROM t ORDER BY a /* sneaky */",
        row_cap=3,
        timeout_s=30,
    )
    assert len(rows) == 3
    assert truncated is True


def test_run_arbitrary_sql_rejects_merge_and_copy(sqlite_adapter):
    """Phase 3 closeout review #5: MERGE/VACUUM/ANALYZE/COPY must trip
    the adapter regex guard, not just the tool-level AST guard."""
    for stmt in (
        "MERGE INTO t USING s ON t.id = s.id WHEN MATCHED THEN DELETE",
        "COPY t FROM '/tmp/x.csv'",
        "VACUUM t",
        "ANALYZE t",
    ):
        with pytest.raises(ValueError, match="read-only"):
            sqlite_adapter.run_arbitrary_sql(stmt, row_cap=500, timeout_s=30)
