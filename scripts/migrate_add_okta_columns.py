"""One-time SQLite migration: add okta_sub and email columns to thrive_user.

Safe to run repeatedly — uses PRAGMA table_info to skip columns that
already exist. Run once on each developer's existing dev DB. New installs
do not need this script (Base.metadata.create_all handles them).

Usage:
    uv run python scripts/migrate_add_okta_columns.py
    uv run python scripts/migrate_add_okta_columns.py --db ./pgDatabase/db.sqlite3
"""

import argparse
import logging
import sqlite3
import sys

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

DEFAULT_DB = "./pgDatabase/db.sqlite3"


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    cursor = conn.execute(f"PRAGMA table_info({table})")
    return any(row[1] == column for row in cursor.fetchall())


def migrate(db_path: str) -> None:
    logger.info("Migrating %s", db_path)
    with sqlite3.connect(db_path) as conn:
        if _column_exists(conn, "thrive_user", "okta_sub"):
            logger.info("okta_sub already present — skipping")
        else:
            conn.execute("ALTER TABLE thrive_user ADD COLUMN okta_sub VARCHAR(255)")
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS ix_thrive_user_okta_sub ON thrive_user(okta_sub)")
            logger.info("Added okta_sub column and unique index")

        if _column_exists(conn, "thrive_user", "email"):
            logger.info("email already present — skipping")
        else:
            conn.execute("ALTER TABLE thrive_user ADD COLUMN email VARCHAR(320) COLLATE NOCASE")
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS ix_thrive_user_email ON thrive_user(email)")
            logger.info("Added email column and unique index")

        # SQLite cannot drop NOT NULL via ALTER TABLE. The original password
        # column was created NOT NULL; new fresh installs via Base.metadata.create_all
        # will use the relaxed nullability. Existing dev DBs continue to enforce
        # NOT NULL on password — acceptable because OIDC users won't be inserted
        # into pre-existing dev DBs without going through the new code path.
        # If you need OIDC users in an existing dev DB, supply a placeholder
        # password (e.g., "OIDC_USER") at insert time.
        conn.commit()
    logger.info("Migration complete")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=DEFAULT_DB, help="Path to SQLite DB (default: %(default)s)")
    args = parser.parse_args()
    try:
        migrate(args.db)
        return 0
    except Exception as exc:
        logger.error("Migration failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
