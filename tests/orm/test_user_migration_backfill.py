"""Backfill verification for Alembic revision 7b3a1f0c92d4 — Epic #179.

Seeds a SQLite DB at the prior head with rows that have NULL ``email``,
NULL ``organization``, and NULL ``user_role_id``, runs the migration to
head, and asserts:

  - the three columns end up NOT NULL on the table definition,
  - the synthetic email values follow the documented sentinel pattern
    ``<missing-email-{id}@unknown.local>``,
  - the organization values are filled with the documented ``"Unknown"`` sentinel,
  - the user_role_id values resolve to the PATIENT role row.
"""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _prior_head() -> str:
    """The revision immediately before 7b3a1f0c92d4."""
    return "188ab391e291"


def test_migration_backfills_null_columns_then_applies_not_null(tmp_path):
    url = f"sqlite:///{tmp_path / 'thrive.sqlite3'}"
    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", url)

    # 1. Migrate to the revision just before Epic #179. Schema still allows
    #    NULL email, organization, user_role_id.
    command.upgrade(cfg, _prior_head())

    engine = create_engine(url)

    # 2. Seed: a PATIENT role row + three users with various NULL fields.
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO thrive_user_role (id, role_name, description, role) "
                "VALUES (4, 'Patient', 'Patient access', 'PATIENT')"
            )
        )
        # User 1: NULL email, NULL organization, NULL user_role_id (all three).
        conn.execute(
            text(
                "INSERT INTO thrive_user (id, username, first_name, last_name, password) "
                "VALUES (1, 'legacy-1', 'L', 'One', 'x')"
            )
        )
        # User 2: real email + role, NULL organization only.
        conn.execute(
            text(
                "INSERT INTO thrive_user (id, username, first_name, last_name, password, "
                "email, user_role_id) "
                "VALUES (2, 'legacy-2', 'L', 'Two', 'x', 'two@real.com', 4)"
            )
        )
        # User 3: real organization + role, NULL email only.
        conn.execute(
            text(
                "INSERT INTO thrive_user (id, username, first_name, last_name, password, "
                "organization, user_role_id) "
                "VALUES (3, 'legacy-3', 'L', 'Three', 'x', 'RealOrg', 4)"
            )
        )

    # 3. Apply Epic #179 migration.
    command.upgrade(cfg, "head")

    # 4. Verify column nullability flipped to NOT NULL.
    inspector = inspect(engine)
    cols = {c["name"]: c for c in inspector.get_columns("thrive_user")}
    assert cols["email"]["nullable"] is False
    assert cols["organization"]["nullable"] is False
    assert cols["user_role_id"]["nullable"] is False

    # 5. Verify backfilled values match the sentinel patterns documented
    #    in the migration docstring.
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT id, email, organization, user_role_id FROM thrive_user ORDER BY id")
        ).fetchall()
        rows_by_id = {r.id: r for r in rows}

        # User 1 — all three sentinels.
        assert rows_by_id[1].email == "<missing-email-1@unknown.local>"
        assert rows_by_id[1].organization == "Unknown"
        assert rows_by_id[1].user_role_id == 4  # PATIENT

        # User 2 — only organization backfilled; email + role untouched.
        assert rows_by_id[2].email == "two@real.com"
        assert rows_by_id[2].organization == "Unknown"
        assert rows_by_id[2].user_role_id == 4

        # User 3 — only email backfilled; organization + role untouched.
        assert rows_by_id[3].email == "<missing-email-3@unknown.local>"
        assert rows_by_id[3].organization == "RealOrg"
        assert rows_by_id[3].user_role_id == 4


def test_migration_downgrade_relaxes_not_null(tmp_path):
    """Downgrade must restore nullability so the migration is reversible.
    Sentinel values stay in place — they're explicitly traceable post-revert."""
    url = f"sqlite:///{tmp_path / 'thrive.sqlite3'}"
    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", url)

    command.upgrade(cfg, _prior_head())

    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO thrive_user_role (id, role_name, description, role) "
                "VALUES (4, 'Patient', 'Patient access', 'PATIENT')"
            )
        )
        conn.execute(
            text(
                "INSERT INTO thrive_user (id, username, first_name, last_name, password, user_role_id) "
                "VALUES (1, 'legacy', 'Leg', 'Acy', 'x', 4)"
            )
        )

    command.upgrade(cfg, "head")

    # Verify it's NOT NULL after upgrade.
    inspector = inspect(engine)
    cols = {c["name"]: c for c in inspector.get_columns("thrive_user")}
    assert cols["email"]["nullable"] is False

    # Downgrade.
    command.downgrade(cfg, _prior_head())

    inspector = inspect(engine)
    cols = {c["name"]: c for c in inspector.get_columns("thrive_user")}
    assert cols["email"]["nullable"] is True
    assert cols["organization"]["nullable"] is True
    assert cols["user_role_id"]["nullable"] is True
