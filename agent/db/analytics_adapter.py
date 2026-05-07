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


_FORBIDDEN_RE = re.compile(
    r"\b(?:INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|GRANT|REVOKE|REPLACE)\b",
    re.IGNORECASE,
)


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

    def fetch_all(self, sql: str, params: Optional[dict] = None) -> list[dict]:
        if _FORBIDDEN_RE.search(sql):
            raise ValueError("Adapter is read-only; statement contains write keyword.")
        with self.engine.connect() as conn:
            result = conn.execute(text(sql), params or {})
            return [dict(row._mapping) for row in result]

    @classmethod
    def from_streamlit_secrets(cls) -> "AnalyticsDbAdapter":
        import streamlit as st
        from sqlalchemy import create_engine

        analytics = dict(st.secrets.get("analytics_db", {}))
        dialect = analytics.get("dialect", "postgres")
        url = analytics.get("url")
        if not url:
            raise RuntimeError("secrets.analytics_db.url is required (postgres:// or redshift://)")
        engine = create_engine(url)
        schema = analytics.get("schema", "") or ""
        return cls(engine=engine, dialect=dialect, schema=schema)
