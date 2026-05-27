"""AnalyticsDbAdapter accumulates the SQL it runs so the agent runtime
can surface it back to the user (gated by role). One log per adapter
instance, popped per tool invocation by the runner.
"""

from __future__ import annotations

from agent.db.analytics_adapter import AnalyticsDbAdapter


def test_fetch_all_appends_to_sql_log(synthetic_db):
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    adapter.fetch_all(
        "SELECT first_name FROM internal_patient_profile_v WHERE last_name = :ln",
        {"ln": "Smith"},
    )
    assert len(adapter.sql_log) == 1
    entry = adapter.sql_log[0]
    assert "internal_patient_profile_v" in entry["sql"]
    assert entry["params"] == {"ln": "Smith"}


def test_pop_sql_log_returns_and_clears(synthetic_db):
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    adapter.fetch_all("SELECT 1 AS one")
    adapter.fetch_all("SELECT 2 AS two")
    popped = adapter.pop_sql_log()
    assert len(popped) == 2
    assert "SELECT 1 AS one" in popped[0]["sql"]
    assert "SELECT 2 AS two" in popped[1]["sql"]
    # Subsequent pop is empty until next fetch_all.
    assert adapter.pop_sql_log() == []


def test_sql_log_starts_empty(synthetic_db):
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    assert adapter.sql_log == []


def test_pop_does_not_mutate_a_returned_snapshot(synthetic_db):
    """The returned list must be a snapshot — the caller can hold onto it
    after the adapter is reused for the next tool."""
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    adapter.fetch_all("SELECT 1 AS one")
    snapshot = adapter.pop_sql_log()
    adapter.fetch_all("SELECT 2 AS two")
    # Snapshot must still reflect only the first call.
    assert len(snapshot) == 1
    assert "SELECT 1" in snapshot[0]["sql"]
