import pytest
from sqlalchemy import text
from agent.db.analytics_adapter import AnalyticsDbAdapter


def test_adapter_wraps_engine(synthetic_db):
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    rows = adapter.fetch_all("SELECT first_name FROM internal_patient_profile_v ORDER BY patient_id")
    assert [r["first_name"] for r in rows] == ["John", "John", "Jane"]


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
