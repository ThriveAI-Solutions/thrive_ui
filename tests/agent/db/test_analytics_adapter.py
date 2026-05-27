import pytest
from sqlalchemy import text
from agent.db.analytics_adapter import AnalyticsDbAdapter


def test_adapter_wraps_engine(synthetic_db):
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    rows = adapter.fetch_all("SELECT first_name FROM internal_patient_profile_v ORDER BY patient_id")
    # Original 3 patients are John, John, Jane; fixture may have more rows.
    assert [r["first_name"] for r in rows[:3]] == ["John", "John", "Jane"]
    assert len(rows) >= 3


def test_adapter_param_binding(synthetic_db):
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    rows = adapter.fetch_all(
        "SELECT first_name FROM internal_patient_profile_v WHERE last_name = :ln",
        {"ln": "Smith"},
    )
    assert len(rows) == 3


def test_adapter_readonly_blocks_write(synthetic_db):
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    with pytest.raises(ValueError, match="read-only"):
        adapter.fetch_all("DELETE FROM internal_patient_profile_v")


def test_adapter_sqlite_engine_rejects_write_inside_explicit_transaction(tmp_path):
    """Defense-in-depth: even when a caller uses engine.begin() (which
    commits DML), SQLite's PRAGMA query_only must reject writes after
    the adapter wraps the engine.

    Uses a file-backed SQLite to mirror production; the synthetic_db
    fixture uses :memory: which has a single-connection lifecycle that
    interferes with connect-event setup.
    """
    from sqlalchemy import create_engine, text as _text

    db_file = tmp_path / "ro.sqlite3"
    setup_engine = create_engine(f"sqlite:///{db_file}")
    with setup_engine.begin() as conn:
        conn.execute(_text("CREATE TABLE t (id INTEGER, val TEXT)"))
        conn.execute(_text("INSERT INTO t VALUES (1, 'a')"))
    setup_engine.dispose()

    # Re-open as the adapter would, then expect read-only enforcement.
    engine = create_engine(f"sqlite:///{db_file}")
    AnalyticsDbAdapter(engine=engine, dialect="sqlite")  # installs pragma hook
    with pytest.raises(Exception):
        with engine.begin() as conn:
            conn.execute(_text("DELETE FROM t"))

    # Confirm the row is still there.
    with engine.connect() as conn:
        rows = conn.execute(_text("SELECT COUNT(*) FROM t")).scalar()
    assert rows == 1
