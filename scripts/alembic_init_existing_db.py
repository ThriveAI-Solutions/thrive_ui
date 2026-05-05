"""One-time helper to bring a pre-Alembic SQLite DB under Alembic management.

Decision tree:
  - alembic_version table exists  -> already initialized, exit 0
  - DB has no tables               -> nothing to stamp; just run the app, exit 0
  - DB has tables, schema matches  -> stamp at baseline, exit 0
  - DB has tables, schema diverges -> print diff, exit 1 (manual reconcile)

Usage:
    uv run python scripts/alembic_init_existing_db.py

Honors THRIVE_SQLITE_PATH env var if set; otherwise reads the path from
Streamlit secrets just like the app.
"""

from __future__ import annotations

import sys
from pathlib import Path

from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import create_engine, inspect, text

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from orm.models import Base, _get_database_url  # noqa: E402


def main() -> int:
    url = _get_database_url()
    print(f"Target DB: {url}")

    engine = create_engine(url)
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())

    if "alembic_version" in tables:
        with engine.connect() as conn:
            row = conn.execute(text("SELECT version_num FROM alembic_version LIMIT 1")).fetchone()
        if row:
            print(f"DB is already under Alembic management at revision {row[0]}. Nothing to do.")
            return 0
        print("alembic_version table exists but is empty — treating as uninitialized and re-checking schema.")
        # Fall through to schema comparison below.

    elif not tables:
        print("DB is empty. Just start the app — init_db() will run the baseline.")
        return 0

    # DB has tables but no version table. Compare schema vs models before stamping.
    with engine.connect() as conn:
        ctx = MigrationContext.configure(conn, opts={"render_as_batch": True})
        diff = compare_metadata(ctx, Base.metadata)

    if diff:
        print("\nSchema does NOT match orm/models.py — refusing to stamp.\n")
        print("Differences detected:")
        for entry in diff:
            print(f"  {entry}")
        print(
            "\nReconcile manually (drop/add columns, fix types) so your schema "
            "matches the models, then re-run this script. Or, if you'd rather "
            "start fresh, delete the SQLite file and let the app recreate it."
        )
        return 1

    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", url)
    command.stamp(cfg, "head")
    print("Stamped at head. Future schema changes go through `alembic revision`.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
