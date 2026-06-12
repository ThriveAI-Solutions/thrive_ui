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


def _capture_engine_kwargs(dialect: str, url: str, monkeypatch) -> dict:
    """Patch create_engine + streamlit.secrets to capture how
    from_streamlit_secrets configures the SQLAlchemy engine for `dialect`."""
    import sqlalchemy as _sqla

    real_create_engine = _sqla.create_engine  # bind BEFORE patching to avoid recursion
    captured: dict = {}

    def fake_create_engine(passed_url, **kwargs):
        captured["url"] = passed_url
        captured["kwargs"] = kwargs
        # Return a SQLite engine so AnalyticsDbAdapter.__post_init__ can
        # install its guards without needing a live Postgres/Redshift.
        return real_create_engine("sqlite:///:memory:")

    fake_secrets = {"analytics_db": {"dialect": dialect, "url": url, "schema": "dw"}}

    class _FakeSt:
        secrets = fake_secrets

    monkeypatch.setattr("sqlalchemy.create_engine", fake_create_engine)
    import sys

    monkeypatch.setitem(sys.modules, "streamlit", _FakeSt)

    # Force the sqlite branch in __post_init__ regardless of the captured
    # dialect — we're only testing how from_streamlit_secrets wires the pool,
    # not the dialect-specific session guards (covered elsewhere). Without
    # this, dialect="postgres" would try SET SESSION CHARACTERISTICS against
    # our in-memory SQLite stand-in.
    original_init = AnalyticsDbAdapter.__post_init__

    def fake_init(self):
        self.dialect = "sqlite"
        original_init(self)

    monkeypatch.setattr(AnalyticsDbAdapter, "__post_init__", fake_init)

    AnalyticsDbAdapter.from_streamlit_secrets()
    return captured


def test_from_streamlit_secrets_enables_pool_pre_ping_for_redshift(monkeypatch):
    """Regression: Redshift idle-killed sockets must be detected before use,
    or the first SET statement_timeout after pool checkout dies with
    'SSL connection has been closed unexpectedly'."""
    captured = _capture_engine_kwargs(
        "redshift", "redshift+psycopg2://u:p@h:5439/db", monkeypatch
    )
    assert captured["kwargs"].get("pool_pre_ping") is True
    assert captured["kwargs"].get("pool_recycle") == 1800


def test_from_streamlit_secrets_enables_pool_pre_ping_for_postgres(monkeypatch):
    captured = _capture_engine_kwargs(
        "postgres", "postgresql+psycopg2://u:p@h:5432/db", monkeypatch
    )
    assert captured["kwargs"].get("pool_pre_ping") is True
    assert captured["kwargs"].get("pool_recycle") == 1800


def test_from_streamlit_secrets_skips_pool_kwargs_for_sqlite(monkeypatch):
    """SQLite has no remote connection to drop; pool flags would be noise."""
    captured = _capture_engine_kwargs("sqlite", "sqlite:///:memory:", monkeypatch)
    assert "pool_pre_ping" not in captured["kwargs"]
    assert "pool_recycle" not in captured["kwargs"]
