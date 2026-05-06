"""Seed/refresh the local SQLite analytics DB used in dev.

Reads the synthetic Redshift mirror SQL at
``tests/agent/redshift_synthetic.sql`` and writes it into a target
SQLite file (default: ``pgDatabase/analytics.sqlite3``).

Run:
    uv run python scripts/init_analytics_db.py

The synthetic SQL begins with ``DROP TABLE IF EXISTS`` for every table,
so re-running this script reseeds rather than appending.

Production points ``[analytics_db]`` at Redshift; this script is a no-op
there because it only writes to a local SQLite file.
"""

from __future__ import annotations
import sys
from pathlib import Path

from sqlalchemy import create_engine, text

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_TARGET = _REPO_ROOT / "pgDatabase" / "analytics.sqlite3"
_SCHEMA_SQL = _REPO_ROOT / "tests" / "agent" / "redshift_synthetic.sql"


def init_analytics_db(target: Path = _DEFAULT_TARGET, schema_sql: Path = _SCHEMA_SQL) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    sql = schema_sql.read_text()
    engine = create_engine(f"sqlite:///{target}")
    with engine.begin() as conn:
        for stmt in sql.split(";"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(text(stmt))
    engine.dispose()


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT_TARGET
    init_analytics_db(target)
    print(f"Initialized analytics SQLite DB at {target}")
