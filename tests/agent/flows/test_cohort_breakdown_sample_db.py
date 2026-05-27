"""sample_db integration: monthly diagnosis breakdown over the loaded
synthetic ~200-patient database.

Requires:
- Docker Postgres running (docker compose up -d)
- Sample dump loaded (uv run python scripts/load_sample_db.py)
- [postgres] in .streamlit/secrets.toml pointing at it

Run: uv run pytest -m sample_db

Without a reachable / loaded Postgres these tests SKIP rather than fail.
"""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest
import sqlalchemy as sa

from agent.db.queries.cohort_breakdown import BreakdownDimension, cohort_breakdown_sql

pytestmark = pytest.mark.sample_db

# The sample dump loads its views into the "dw" schema (see scripts/load_sample_db.py
# and data/sample/thrive_sample.sql.zst), and Postgres is the loaded dialect.
SCHEMA_PREFIX = "dw."
DIALECT = "postgres"


def _engine_or_skip() -> sa.Engine:
    """Build a Postgres engine and probe it; skip the test if unreachable.

    Mirrors tests/sample_db/test_agent_on_sample_db.py — the established
    sample_db connection mechanism in this repo (no shared fixture exists).
    """
    import tomllib

    try:
        pg = tomllib.loads(open(".streamlit/secrets.toml").read())["postgres"]
    except (FileNotFoundError, KeyError) as e:
        pytest.skip(f"Postgres config not found: {e}")

    url = f"postgresql+psycopg2://{pg['user']}:{pg['password']}@{pg['host']}:{pg['port']}/{pg['database']}"
    eng = sa.create_engine(url)
    try:
        with eng.connect() as conn:
            conn.execute(sa.text("SELECT 1"))
            # internal_source_reference_v is created exclusively by
            # load_sample_db.py, so its absence means the dump isn't loaded.
            table_exists = conn.execute(
                sa.text(
                    """
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_schema = 'dw'
                      AND table_name = 'internal_source_reference_v'
                    """
                )
            ).scalar()
            if not table_exists:
                pytest.skip("Sample DB tables not found — run 'uv run python scripts/load_sample_db.py' first")
    except pytest.skip.Exception:
        raise
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Postgres sample DB not reachable: {e}")
    return eng


def _crit(**kw):
    base = dict(
        diagnosis_codes=None,
        diagnosis_date_range=None,
        medication_rxnorm_codes=None,
        condition_text=None,
        age_min=None,
        age_max=None,
        gender=None,
        facility=None,
        last_visit_after=None,
        last_visit_before=None,
        zip_code=None,
        city=None,
        state=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_monthly_breakdown_runs_on_sample_db():
    """The diagnosis-month breakdown SQL executes end-to-end against the
    loaded sample DB and returns sane monthly buckets."""
    eng = _engine_or_skip()

    # A wide window that the sample diagnosis data populates densely.
    crit = _crit(diagnosis_date_range=SimpleNamespace(start=date(2020, 1, 1), end=date(2026, 12, 31)))

    bucket_sql, total_sql, params = cohort_breakdown_sql(
        crit,
        BreakdownDimension.DIAGNOSIS_MONTH,
        schema_prefix=SCHEMA_PREFIX,
        dialect=DIALECT,
    )

    with eng.connect() as conn:
        buckets = conn.execute(sa.text(bucket_sql), params).mappings().all()
        total = conn.execute(sa.text(total_sql), params).scalar()

    assert buckets, "expected at least one monthly bucket from the sample DB"
    for row in buckets:
        assert row["patient_count"] >= 1, f"bucket {row['bucket_label']!r} has count {row['patient_count']}"
    assert total is not None and total >= 1, f"expected total_count >= 1, got {total}"
