"""Connection adapter for the analytics database.

Routes one of: SQLite (test/dev), Postgres (dev), Redshift (prod), or
Amazon RDS Postgres. The dialect string lets callers detect which
engine-specific quirks apply (e.g., Redshift uses pg_views, lacks
some Postgres functions).

Per spec §7.7 the adapter is read-only at the SQLAlchemy level.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.engine import Engine


_FORBIDDEN_RE = re.compile(
    r"\b(?:INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|GRANT|REVOKE|REPLACE)\b",
    re.IGNORECASE,
)


@dataclass
class AnalyticsDbAdapter:
    engine: Engine
    dialect: str  # "sqlite" | "postgres" | "redshift"

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
        engine = create_engine(url, execution_options={"readonly": True})
        return cls(engine=engine, dialect=dialect)
