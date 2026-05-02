"""One-time SQLite migration: Okta columns + wider ``username`` on ``thrive_user``.

- Adds ``okta_sub`` and ``email`` if missing (idempotent).
- If ``username`` is still compiled as VARCHAR(50) from older ORM schemas, replaces
  the table via rename + recreated DDL (:func:`sqlite3` cannot ``ALTER``
  widening on a UNIQUE column in place).

Safe to run repeatedly — uses PRAGMA table_info to skip columns that
already exist. New installs do not need this script (``Base.metadata.create_all``
matches the ORM).

Usage:
    uv run python scripts/migrate_add_okta_columns.py
    uv run python scripts/migrate_add_okta_columns.py --db ./pgDatabase/db.sqlite3
"""

import argparse
import logging
import re
import sqlite3
import sys

# Backup table name while rebuilding ``thrive_user`` for VARCHAR(320) username.
_USERNAME_REBUILD_TMP = "_thrive_user__okta_username_wide_backup"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

DEFAULT_DB = "./pgDatabase/db.sqlite3"


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    cursor = conn.execute(f"PRAGMA table_info({table})")
    return any(row[1] == column for row in cursor.fetchall())


def _username_sqlite_declared_type(conn: sqlite3.Connection) -> str | None:
    for row in conn.execute("PRAGMA table_info(thrive_user)"):
        if row[1] == "username":
            return row[2] if row[2] else None
    return None


def _username_needs_varchar_wide_migration(typ: str | None) -> bool:
    """True when affinity looks VARCHAR(50) / STRING(50) (legacy ORM width).

    Matches whole ``(50)`` width modifier, so ``VARCHAR(250)`` does not falsely match.
    """
    if not typ or not typ.strip():
        return False
    s = typ.strip()
    pattern = (
        r"\b(?:VARCHAR|STRING|CHAR\s+VARYING)\s*\(\s*50\s*\)"
        r"|\bCHAR\s*\(\s*50\s*\)"
    )
    return bool(re.search(pattern, s, flags=re.IGNORECASE))


def _standalone_index_ddls(conn: sqlite3.Connection, table: str) -> list[str]:
    rows = conn.execute(
        """
        SELECT sql FROM sqlite_master
        WHERE type = 'index'
          AND tbl_name = ?
          AND sql IS NOT NULL
          AND name NOT LIKE 'sqlite_autoindex%%'
        """,
        (table,),
    ).fetchall()
    return [r[0] for r in rows if r[0]]


def _widen_username_recreate_table(conn: sqlite3.Connection) -> None:
    """Match ORM VARCHAR(320) ``username`` for OIDC JIT (email-sized usernames).

    SQLite cannot widen a UNIQUE column with a one-liner ALTER. We snapshot
    ``CREATE TABLE`` from ``sqlite_master``, rewrite ``username``, copy rows,
    and re-apply non-auto indexes.
    """
    if not _column_exists(conn, "thrive_user", "username"):
        return

    pragma_type = _username_sqlite_declared_type(conn)
    if not _username_needs_varchar_wide_migration(pragma_type):
        logger.debug("username SQLite type %r — skip VARCHAR(320) rebuild", pragma_type)
        return

    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
        ("thrive_user",),
    ).fetchone()
    if not row or not row[0]:
        logger.warning("No CREATE TABLE DDL for thrive_user in sqlite_master; skip username widen")
        return

    ddl = row[0]
    new_ddl = re.sub(
        r"(\busername\b\s+VARCHAR\s*\()\s*50\s*(\))",
        lambda m: f"{m.group(1)}320{m.group(2)}",
        ddl,
        count=1,
        flags=re.IGNORECASE,
    )
    if new_ddl == ddl:
        new_ddl = re.sub(
            r"(\busername\b\s+STRING\s*\()\s*50\s*(\))",
            lambda m: f"{m.group(1)}320{m.group(2)}",
            ddl,
            count=1,
            flags=re.IGNORECASE,
        )

    if new_ddl == ddl:
        logger.warning(
            "Could not rewrite username width in DDL (affinity %r) — widen skipped; "
            "SQLite does not enforce VARCHAR length.",
            pragma_type,
        )
        return

    indexes = _standalone_index_ddls(conn, "thrive_user")

    logger.info(
        "Rebuilding thrive_user so username is VARCHAR(320) (SQLite had %r)",
        pragma_type,
    )
    fk_was = conn.execute("PRAGMA foreign_keys").fetchone()[0]
    try:
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute(f'ALTER TABLE thrive_user RENAME TO "{_USERNAME_REBUILD_TMP}"')
        conn.execute(new_ddl)

        colnames = [row[1] for row in conn.execute(f'PRAGMA table_info("{_USERNAME_REBUILD_TMP}")')]
        cols_sql = ", ".join(f'"{c}"' for c in colnames)
        conn.execute(f'INSERT INTO thrive_user ({cols_sql}) SELECT {cols_sql} FROM "{_USERNAME_REBUILD_TMP}"')
        conn.execute(f'DROP TABLE "{_USERNAME_REBUILD_TMP}"')

        for index_sql in indexes:
            conn.execute(index_sql)

    finally:
        conn.execute(f"PRAGMA foreign_keys={fk_was}")
    logger.info("username column rebuild complete (ORM width 320)")


def migrate(db_path: str) -> None:
    logger.info("Migrating %s", db_path)
    with sqlite3.connect(db_path) as conn:
        if _column_exists(conn, "thrive_user", "okta_sub"):
            logger.info("okta_sub already present — skipping")
        else:
            conn.execute("ALTER TABLE thrive_user ADD COLUMN okta_sub VARCHAR(255)")
            logger.info("Added okta_sub column")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS ix_thrive_user_okta_sub ON thrive_user(okta_sub)")
        logger.info("Ensured okta_sub unique index")

        if _column_exists(conn, "thrive_user", "email"):
            logger.info("email already present — skipping")
        else:
            conn.execute("ALTER TABLE thrive_user ADD COLUMN email VARCHAR(320) COLLATE NOCASE")
            logger.info("Added email column")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS ix_thrive_user_email ON thrive_user(email)")
        logger.info("Ensured email unique index")

        _widen_username_recreate_table(conn)

        # The password column remains NOT NULL. OIDC-only users are inserted
        # with a reserved non-hash sentinel by utils.okta_auth so migrated
        # SQLite databases and fresh schemas behave the same way.
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
