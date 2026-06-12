"""Connection adapter for the analytics database.

Routes one of: SQLite (test/dev), Postgres (dev), Redshift (prod), or
Amazon RDS Postgres. The dialect string lets callers detect which
engine-specific quirks apply (e.g., Redshift uses pg_views, lacks
some Postgres functions).

Per spec §7.7 the adapter is read-only. Three layers of enforcement:
1. _FORBIDDEN_RE rejects obvious write keywords before they hit the DB.
2. SQLite: PRAGMA query_only = ON installed via connect event listener.
3. Postgres/Redshift: default_transaction_read_only=on at session level.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from sqlalchemy import event, text
from sqlalchemy.engine import Engine


# Mirror agent.tools.run_sql._FORBIDDEN_KEYWORDS. The two lists must stay
# in sync: this adapter-level regex is the belt-and-suspenders backstop
# for the tool-level AST guard, and a gap on either side defeats both.
_FORBIDDEN_RE = re.compile(
    r"\b(?:INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE"
    r"|GRANT|REVOKE|REPLACE|MERGE|VACUUM|ANALYZE|COPY)\b",
    re.IGNORECASE,
)


def _strip_trailing_sql_comments(sql: str) -> str:
    """Strip trailing whitespace, semicolons, and SQL comments.

    Without this, an attacker (or a careless caller) can defeat
    LIMIT injection by ending the query with `-- anything`: the
    regex below would see no LIMIT, append ` LIMIT 501`, and the
    appended text would land inside the comment — leaving the
    query effectively un-capped.
    """
    prev = None
    while sql != prev:
        prev = sql
        sql = sql.rstrip().rstrip(";").rstrip()
        # Trailing /* ... */ block comment (may span lines).
        sql = re.sub(r"/\*.*?\*/\s*\Z", "", sql, flags=re.DOTALL)
        # Trailing -- line comment (anchored to end of string).
        sql = re.sub(r"--[^\n]*\Z", "", sql)
    return sql


def _inject_limit(sql: str, row_cap: int) -> str:
    """Return SQL bounded by min(existing LIMIT, row_cap+1).

    The +1 is a sentinel for truncation detection. The caller trims
    the extra row and flips `truncated=True`.

    If the SQL already has a LIMIT smaller than row_cap, leave it alone
    (the user explicitly asked for fewer rows; respect that).

    Known limitation: queries that end in `LIMIT N OFFSET M` aren't
    recognized as already-limited and will get a second `LIMIT row_cap+1`
    appended, producing a syntax error. `run_sql` callers are guided
    away from OFFSET in the system prompt; revisit if it becomes a real
    pattern.
    """
    stripped = _strip_trailing_sql_comments(sql)
    limit_match = re.search(r"\blimit\b\s+(\d+)\s*$", stripped, re.IGNORECASE)
    if limit_match:
        user_limit = int(limit_match.group(1))
        if user_limit <= row_cap:
            return stripped
        # User wrote LIMIT 9999 but our cap is 500 — replace with cap+1
        return re.sub(
            r"\blimit\b\s+\d+\s*$",
            f"LIMIT {row_cap + 1}",
            stripped,
            flags=re.IGNORECASE,
        )
    return f"{stripped} LIMIT {row_cap + 1}"


def _install_sqlite_query_only(engine: Engine) -> None:
    @event.listens_for(engine, "connect")
    def _set_query_only(dbapi_connection, connection_record):  # noqa: ARG001
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA query_only = ON")
        cursor.close()

    # Apply to any already-pooled connections so the guard activates
    # immediately, not only on the next checkout.
    with engine.connect() as conn:
        conn.exec_driver_sql("PRAGMA query_only = ON")


def _install_pg_read_only(engine: Engine) -> None:
    @event.listens_for(engine, "connect")
    def _set_read_only(dbapi_connection, connection_record):  # noqa: ARG001
        cursor = dbapi_connection.cursor()
        cursor.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY")
        cursor.close()


@dataclass
class AnalyticsDbAdapter:
    engine: Engine
    dialect: str  # "sqlite" | "postgres" | "redshift"
    schema: str = ""  # e.g. "dw" for prod Redshift; "" for SQLite/search_path-based
    _guards_installed: bool = field(default=False, init=False, repr=False)
    # Per-adapter buffer of executed (sql, params) pairs. The runtime pops
    # this after each tool invocation so the streamed ToolCallCompleted
    # event can carry the SQL that produced the result. Tools run
    # sequentially in the agent loop, so the buffer is unambiguously
    # attributable to the most recent tool call.
    sql_log: list[dict] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if self._guards_installed:
            return
        if self.dialect == "sqlite":
            _install_sqlite_query_only(self.engine)
        elif self.dialect in ("postgres", "redshift"):
            _install_pg_read_only(self.engine)
        self._guards_installed = True

    @property
    def schema_prefix(self) -> str:
        """String to prepend to unqualified table names in raw SQL templates.

        Returns "" when schema is empty (SQLite, or any engine where the
        connection's search_path already includes the target schema), or
        "{schema}." otherwise. Idempotent on a trailing dot the user may
        have written in secrets.toml.
        """
        s = (self.schema or "").rstrip(".")
        return f"{s}." if s else ""

    # Statement-level timeout for the curated query path (cohort, find_patient,
    # get_patient_clinical_data, etc.). Without this, a broad cohort query
    # ('all NY + 3 diabetes codes') ran 13 minutes on the dev warehouse on
    # 2026-05-13 before the runner's wall_clock cap noticed.
    #
    # 240s = wide tolerance for the live HEALTHeLINK Redshift cluster, which
    # under contention will take 30-90s to satisfy queries that complete in
    # <1s when idle. The runner's max_wall_clock_s should be set to ≥300 in
    # secrets.toml so the agent has room to think + answer after a slow query
    # consumes most of this budget.
    _CURATED_QUERY_TIMEOUT_S = 240

    def fetch_all(self, sql: str, params: Optional[dict] = None) -> list[dict]:
        if _FORBIDDEN_RE.search(sql):
            raise ValueError("Adapter is read-only; statement contains write keyword.")
        self.sql_log.append({"sql": sql, "params": dict(params or {})})
        with self.engine.connect() as conn:
            if self.dialect in ("postgres", "redshift"):
                conn.exec_driver_sql(f"SET statement_timeout = {int(self._CURATED_QUERY_TIMEOUT_S * 1000)}")
            result = conn.execute(text(sql), params or {})
            return [dict(row._mapping) for row in result]

    def run_arbitrary_sql(
        self,
        sql: str,
        row_cap: int,
        timeout_s: int,
    ) -> tuple[list[str], list[list], bool]:
        """Execute caller-supplied SQL with row cap + timeout.

        Returns (columns, rows, truncated). Rows is a list of lists
        (positional values matching `columns`), to keep the boundary
        explicit and friendly for JSON serialization downstream.

        Defenses (Phase 3 design §3.4):
          1. _FORBIDDEN_RE rejects writes before the query hits the engine.
          2. LIMIT injection: if there's no LIMIT, append LIMIT row_cap+1;
             otherwise keep the smaller of the user's LIMIT and row_cap.
          3. Statement timeout: SET statement_timeout = <ms> for
             postgres/redshift sessions. SQLite has no equivalent and
             relies on the row cap.
          4. Truncation flag: if row_cap+1 rows were returned, drop the
             last one and mark truncated=True.
        """
        if _FORBIDDEN_RE.search(sql):
            raise ValueError("Adapter is read-only; statement contains write keyword.")

        bounded_sql = _inject_limit(sql, row_cap)
        self.sql_log.append({"sql": bounded_sql, "params": {}})

        with self.engine.connect() as conn:
            if self.dialect in ("postgres", "redshift"):
                conn.exec_driver_sql(f"SET statement_timeout = {int(timeout_s * 1000)}")
            cursor_result = conn.execute(text(bounded_sql))
            columns = list(cursor_result.keys())
            raw_rows = [list(row) for row in cursor_result.fetchall()]

        if len(raw_rows) > row_cap:
            return columns, raw_rows[:row_cap], True
        return columns, raw_rows, False

    def pop_sql_log(self) -> list[dict]:
        """Return the accumulated SQL log and reset it.

        Returns a snapshot (new list); the caller owns it and is unaffected
        by subsequent fetch_all calls on this adapter.
        """
        snapshot = self.sql_log
        self.sql_log = []
        return snapshot

    @classmethod
    def from_streamlit_secrets(cls) -> "AnalyticsDbAdapter":
        import streamlit as st
        from sqlalchemy import create_engine

        analytics = dict(st.secrets.get("analytics_db", {}))
        dialect = analytics.get("dialect", "postgres")
        url = analytics.get("url")
        if not url:
            raise RuntimeError("secrets.analytics_db.url is required (postgres:// or redshift://)")
        # Redshift (and the AWS ELB in front of it) close idle TCP sockets
        # well before SQLAlchemy's pool decides a connection is stale. Without
        # pool_pre_ping we hand the dead socket to the next caller, and the
        # very first statement after checkout — SET statement_timeout in
        # fetch_all/run_arbitrary_sql — raises "SSL connection has been
        # closed unexpectedly". pool_recycle bounds the worst case by
        # forcing periodic reconnects under the typical idle cutoff.
        engine_kwargs: dict = {}
        if dialect in ("postgres", "redshift"):
            engine_kwargs["pool_pre_ping"] = True
            engine_kwargs["pool_recycle"] = 1800
        engine = create_engine(url, **engine_kwargs)
        schema = analytics.get("schema", "") or ""
        return cls(engine=engine, dialect=dialect, schema=schema)
