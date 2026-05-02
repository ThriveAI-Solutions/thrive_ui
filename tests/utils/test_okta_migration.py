"""Tests for the one-time Okta SQLite migration."""

import sqlite3

from scripts.migrate_add_okta_columns import migrate


def test_migration_creates_missing_unique_indexes_when_columns_exist(tmp_path):
    """Re-running after a partial/manual migration should still add indexes."""
    db_path = tmp_path / "db.sqlite3"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE thrive_user (
                id INTEGER PRIMARY KEY,
                password VARCHAR(255) NOT NULL,
                okta_sub VARCHAR(255),
                email VARCHAR(320) COLLATE NOCASE
            )
            """
        )
        conn.commit()

    migrate(str(db_path))

    with sqlite3.connect(db_path) as conn:
        indexes = {row[1] for row in conn.execute("PRAGMA index_list(thrive_user)").fetchall()}

    assert "ix_thrive_user_okta_sub" in indexes
    assert "ix_thrive_user_email" in indexes


def test_migration_rewrites_legacy_username_column_width(tmp_path):
    """VARCHAR(50) ``username`` is rebuilt so OIDC JIT can store longer emails."""
    import re
    import sqlite3

    db_path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE thrive_user (
                id INTEGER PRIMARY KEY,
                username VARCHAR(50) NOT NULL UNIQUE,
                password VARCHAR(255) NOT NULL
            );
            """
        )
        conn.commit()

    def _username_sqlite_decl():
        with sqlite3.connect(db_path) as conn:
            for row in conn.execute("PRAGMA table_info(thrive_user)"):
                if row[1] == "username":
                    return row[2]
        raise AssertionError("username column missing")

    migrate(str(db_path))

    assert re.search(r"VARCHAR\s*\(\s*320\s*\)", _username_sqlite_decl(), flags=re.I)

    migrate(str(db_path))
    assert re.search(r"VARCHAR\s*\(\s*320\s*\)", _username_sqlite_decl(), flags=re.I)
