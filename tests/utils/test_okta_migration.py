"""Tests for the one-time Okta SQLite migration."""

import sqlite3

from scripts.legacy_migrations.migrate_add_okta_columns import migrate


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


def test_migration_username_widen_rolls_back_on_crash(tmp_path, monkeypatch):
    """Crash mid-rebuild leaves thrive_user intact and recoverable on re-run.

    Simulates a SIGKILL or unexpected exception between the table rename and
    the index re-apply step. The DB must end up in a consistent state — either
    fully migrated or fully unmigrated — never with thrive_user missing.
    """
    import re
    import sqlite3

    from scripts.legacy_migrations import migrate_add_okta_columns

    db_path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE thrive_user (
                id INTEGER PRIMARY KEY,
                username VARCHAR(50) NOT NULL UNIQUE,
                password VARCHAR(255) NOT NULL
            );
            INSERT INTO thrive_user (username, password) VALUES ('alice', 'pw1');
            INSERT INTO thrive_user (username, password) VALUES ('bob', 'pw2');
            """
        )
        conn.commit()

    # Inject a crash inside the rebuild's critical section, after the rename
    # has happened but before the new table is fully populated and the
    # backup is dropped.
    real_apply = migrate_add_okta_columns._apply_indexes

    def boom(*args, **kwargs):
        raise RuntimeError("simulated crash mid-rebuild")

    monkeypatch.setattr(migrate_add_okta_columns, "_apply_indexes", boom)

    try:
        migrate_add_okta_columns.migrate(str(db_path))
    except RuntimeError as exc:
        assert "simulated crash" in str(exc)
    else:
        raise AssertionError("Migration should have raised mid-rebuild")

    # After the crash, thrive_user must still exist with all original rows
    # and the backup table must be gone.
    with sqlite3.connect(db_path) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "thrive_user" in tables, f"thrive_user missing after crash: {tables}"
        assert "_thrive_user__okta_username_wide_backup" not in tables, f"backup table not rolled back: {tables}"
        usernames = {row[0] for row in conn.execute("SELECT username FROM thrive_user")}
        assert usernames == {"alice", "bob"}

    # Restore real helper and re-run — must complete successfully.
    monkeypatch.setattr(migrate_add_okta_columns, "_apply_indexes", real_apply)
    migrate_add_okta_columns.migrate(str(db_path))

    with sqlite3.connect(db_path) as conn:
        for row in conn.execute("PRAGMA table_info(thrive_user)"):
            if row[1] == "username":
                assert re.search(r"VARCHAR\s*\(\s*320\s*\)", row[2], flags=re.I)
                break
        else:
            raise AssertionError("username column missing after recovery run")
        usernames = {row[0] for row in conn.execute("SELECT username FROM thrive_user")}
        assert usernames == {"alice", "bob"}
