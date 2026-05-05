"""Migration suite — verifies the Alembic baseline matches `orm/models.py`."""

from __future__ import annotations

from pathlib import Path

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import create_engine, inspect

from orm.models import Base


REPO_ROOT = Path(__file__).resolve().parent.parent

EXPECTED_TABLES = {
    "thrive_user_role",
    "thrive_user",
    "thrive_message",
    "thrive_user_activity",
    "thrive_admin_action",
    "thrive_llm_context",
    "thrive_error_log",
}


@pytest.fixture
def alembic_cfg(tmp_path):
    """Alembic Config pointed at a fresh tmp SQLite DB."""
    db_path = tmp_path / "thrive.sqlite3"
    url = f"sqlite:///{db_path}"
    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", url)
    return cfg, url


def test_upgrade_head_creates_full_schema(alembic_cfg):
    cfg, url = alembic_cfg
    command.upgrade(cfg, "head")

    inspector = inspect(create_engine(url))
    tables = set(inspector.get_table_names())
    assert EXPECTED_TABLES.issubset(tables), f"missing tables: {EXPECTED_TABLES - tables}"
    assert "alembic_version" in tables

    message_indexes = {ix["name"] for ix in inspector.get_indexes("thrive_message")}
    assert "ix_thrive_message_user_id" in message_indexes
    assert "ix_thrive_message_created_at" in message_indexes


def test_upgrade_matches_models_metadata(alembic_cfg):
    """Drift sentinel: models edited without a revision should fail this test.

    Caveats: `compare_metadata` does NOT catch enum value additions on SQLite
    (alembic#329) and by default does not compare server_defaults or column
    type lengths. Adding a new RoleTypeEnum member without a revision will
    pass this test silently — autogenerate against a fresh DB is the broader
    check.
    """
    cfg, url = alembic_cfg
    command.upgrade(cfg, "head")

    diff = _diff_models_against_db(url)
    assert diff == [], f"models drifted from migrations: {diff}"


def test_upgrade_is_idempotent(alembic_cfg):
    cfg, url = alembic_cfg
    command.upgrade(cfg, "head")
    head_after_first = _current_revision(url)

    command.upgrade(cfg, "head")
    head_after_second = _current_revision(url)

    assert head_after_first == head_after_second


def test_downgrade_to_base_then_upgrade_roundtrips(alembic_cfg):
    """Catches missing or wrong downgrade() bodies in future revisions."""
    cfg, url = alembic_cfg
    command.upgrade(cfg, "head")
    command.downgrade(cfg, "base")

    inspector = inspect(create_engine(url))
    tables = set(inspector.get_table_names())
    # alembic_version is the only table that should remain at base
    assert tables.isdisjoint(EXPECTED_TABLES)

    command.upgrade(cfg, "head")
    inspector = inspect(create_engine(url))
    assert EXPECTED_TABLES.issubset(set(inspector.get_table_names()))
    # The roundtrip must produce a schema indistinguishable from a clean upgrade.
    assert _diff_models_against_db(url) == []


def _current_revision(url: str) -> str | None:
    with create_engine(url).connect() as conn:
        return MigrationContext.configure(conn).get_current_revision()


def _diff_models_against_db(url: str) -> list:
    with create_engine(url).connect() as conn:
        ctx = MigrationContext.configure(conn, opts={"render_as_batch": True})
        return compare_metadata(ctx, Base.metadata)
