"""Tests for scripts/init_analytics_db.py.

The seed script creates a SQLite analytics DB by running the synthetic
Redshift mirror SQL against a target file. It powers the dev workflow
where [analytics_db] points at a local file rather than a real warehouse.
"""

from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

from scripts.init_analytics_db import init_analytics_db


def test_init_analytics_db_creates_file_with_expected_tables(tmp_path: Path) -> None:
    target = tmp_path / "analytics.sqlite3"

    init_analytics_db(target)

    assert target.exists()
    engine = create_engine(f"sqlite:///{target}")
    with engine.connect() as conn:
        names = {row[0] for row in conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))}
    assert "internal_patient_profile_v" in names
    assert "internal_source_reference_v" in names
    assert "federated_demographic_v" in names
    assert "federated_encounters_v" in names


def test_init_analytics_db_seeds_patient_rows(tmp_path: Path) -> None:
    target = tmp_path / "analytics.sqlite3"

    init_analytics_db(target)

    engine = create_engine(f"sqlite:///{target}")
    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM internal_patient_profile_v")).scalar()
    assert count and count >= 3


def test_init_analytics_db_is_idempotent(tmp_path: Path) -> None:
    target = tmp_path / "analytics.sqlite3"

    init_analytics_db(target)
    init_analytics_db(target)  # second run must not raise

    engine = create_engine(f"sqlite:///{target}")
    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM internal_patient_profile_v")).scalar()
    # idempotent reseed should produce the same row count, not double it
    assert count == 3
