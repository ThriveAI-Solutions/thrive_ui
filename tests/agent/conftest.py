"""Shared fixtures for agent tool tests.

Provides a per-test SQLite engine loaded with the synthetic Redshift
mirror. Tests query against this in place of real Redshift.
"""

from pathlib import Path
import pytest
from sqlalchemy import create_engine, text


_SQL_FILE = Path(__file__).parent / "redshift_synthetic.sql"


@pytest.fixture
def synthetic_db():
    """In-memory SQLite engine pre-loaded with the §7.12 whitelist mirror."""
    engine = create_engine("sqlite:///:memory:")
    sql = _SQL_FILE.read_text()
    with engine.begin() as conn:
        for stmt in sql.split(";"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(text(stmt))
    return engine
