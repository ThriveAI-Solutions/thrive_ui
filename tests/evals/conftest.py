from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

from agent.db.analytics_adapter import AnalyticsDbAdapter

_SYNTHETIC_SQL = Path(__file__).resolve().parents[1] / "agent/redshift_synthetic.sql"


@pytest.fixture
def synthetic_adapter() -> AnalyticsDbAdapter:
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    with engine.begin() as conn:
        for stmt in _SYNTHETIC_SQL.read_text().split(";"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(text(stmt))
    return AnalyticsDbAdapter(engine=engine, dialect="sqlite")
