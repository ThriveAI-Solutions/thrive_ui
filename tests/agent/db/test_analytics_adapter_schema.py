"""AnalyticsDbAdapter must carry an optional schema name and expose a
schema_prefix usable by query templates. Production Redshift puts the
agent's views in a `dw` schema; SQLite has no schema concept, so the
default prefix must be empty.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from sqlalchemy import create_engine

from agent.db.analytics_adapter import AnalyticsDbAdapter


def test_default_schema_is_empty(synthetic_db):
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
    assert adapter.schema == ""
    assert adapter.schema_prefix == ""


def test_schema_prefix_appends_dot(synthetic_db):
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="redshift", schema="dw")
    assert adapter.schema == "dw"
    assert adapter.schema_prefix == "dw."


def test_schema_prefix_normalizes_trailing_dot(synthetic_db):
    """If a user writes `schema = "dw."` in secrets they shouldn't get `dw..`."""
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="redshift", schema="dw.")
    assert adapter.schema_prefix == "dw."


def test_schema_prefix_empty_when_blank_string(synthetic_db):
    adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite", schema="")
    assert adapter.schema_prefix == ""


def test_from_streamlit_secrets_reads_schema():
    """Production wires schema via [analytics_db].schema in secrets.toml."""
    fake_secrets = {
        "analytics_db": {
            "url": "postgresql+psycopg2://u:p@h/db",
            "dialect": "redshift",
            "schema": "dw",
        }
    }
    real_engine = create_engine("sqlite:///:memory:")
    with patch("sqlalchemy.create_engine", return_value=real_engine), patch("streamlit.secrets", fake_secrets):
        adapter = AnalyticsDbAdapter.from_streamlit_secrets()
    assert adapter.dialect == "redshift"
    assert adapter.schema == "dw"
    assert adapter.schema_prefix == "dw."


def test_from_streamlit_secrets_schema_optional():
    """Schema is optional — defaults to empty string for SQLite/local dev."""
    fake_secrets = {
        "analytics_db": {
            "url": "sqlite:///./pgDatabase/analytics.sqlite3",
            "dialect": "sqlite",
        }
    }
    real_engine = create_engine("sqlite:///:memory:")
    with patch("sqlalchemy.create_engine", return_value=real_engine), patch("streamlit.secrets", fake_secrets):
        adapter = AnalyticsDbAdapter.from_streamlit_secrets()
    assert adapter.schema == ""
    assert adapter.schema_prefix == ""
