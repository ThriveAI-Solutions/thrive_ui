"""db_elapsed_ms must appear on every sql_log entry (spec: eval harness §4)."""

import pytest
from sqlalchemy import create_engine

from agent.db.analytics_adapter import AnalyticsDbAdapter


@pytest.fixture
def adapter():
    engine = create_engine("sqlite:///:memory:")
    return AnalyticsDbAdapter(engine=engine, dialect="sqlite")


def test_fetch_all_records_db_elapsed_ms(adapter):
    adapter.fetch_all("SELECT 1 AS x")
    log = adapter.pop_sql_log()
    assert len(log) == 1
    assert isinstance(log[0]["db_elapsed_ms"], int)
    assert log[0]["db_elapsed_ms"] >= 0


def test_run_arbitrary_sql_records_db_elapsed_ms(adapter):
    adapter.run_arbitrary_sql("SELECT 1 AS x", row_cap=10, timeout_s=5)
    log = adapter.pop_sql_log()
    assert len(log) == 1
    assert isinstance(log[0]["db_elapsed_ms"], int)
    assert log[0]["db_elapsed_ms"] >= 0
