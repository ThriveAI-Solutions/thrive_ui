"""show_thinking_process should default OFF for everybody — new users and existing.

- New users created via the ORM / ``create_user`` get ``show_thinking_process = False``.
- The schema migration backfills any pre-existing rows to ``False``.
- ``User.to_dict()`` exposes the new key so downstream serializers see it.
"""

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text

from orm.functions import create_user, update_user_preferences
from orm.models import User, UserRole

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_create_user_defaults_show_thinking_process_off(in_memory_orm_session):
    with in_memory_orm_session() as session:
        role_id = session.query(UserRole).filter(UserRole.role_name == "Doctor").one().id

    assert (
        create_user(
            "thinkdoc",
            "pw",
            "Think",
            "Doc",
            role_id,
            email="thinkdoc@example.com",
            organization="Acme",
        )
        is True
    )

    with in_memory_orm_session() as session:
        user = session.query(User).filter(User.username == "thinkdoc").one()
        assert user.show_thinking_process is False


def test_user_to_dict_includes_show_thinking_process(in_memory_orm_session):
    with in_memory_orm_session() as session:
        role_id = session.query(UserRole).filter(UserRole.role_name == "Nurse").one().id

    create_user(
        "dictnurse",
        "pw",
        "Dict",
        "Nurse",
        role_id,
        email="dictnurse@example.com",
        organization="Acme",
    )

    with in_memory_orm_session() as session:
        user = session.query(User).filter(User.username == "dictnurse").one()
        payload = user.to_dict()
        assert "show_thinking_process" in payload
        assert payload["show_thinking_process"] is False


def test_update_user_preferences_persists_show_thinking_process(in_memory_orm_session):
    """The admin-capable ``update_user_preferences`` path must accept
    ``show_thinking_process`` — exercises the allowlist + round-trip
    persistence, since the self-service ``save_user_settings`` path is
    Streamlit-coupled and harder to unit-test directly."""
    with in_memory_orm_session() as session:
        role_id = session.query(UserRole).filter(UserRole.role_name == "Doctor").one().id

    create_user(
        "rtuser",
        "pw",
        "Round",
        "Trip",
        role_id,
        email="rtuser@example.com",
        organization="Acme",
    )

    with in_memory_orm_session() as session:
        user = session.query(User).filter(User.username == "rtuser").one()
        assert user.show_thinking_process is False
        user_id = user.id

    assert update_user_preferences(user_id, show_thinking_process=True) is True

    with in_memory_orm_session() as session:
        user = session.query(User).filter(User.id == user_id).one()
        assert user.show_thinking_process is True


def test_migration_backfills_show_thinking_process_to_false(tmp_path):
    """Upgrade to the prior revision, insert a user without the new column,
    then upgrade to head — the new migration must add the column and backfill
    every row to ``0`` (False)."""
    url = f"sqlite:///{tmp_path / 'thrive.sqlite3'}"
    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", url)

    # Schema state just before the show_thinking_process migration.
    command.upgrade(cfg, "c8d5e3a90212")

    engine = create_engine(url)
    with engine.begin() as conn:
        # Seed a minimal UserRole row so the Epic #179 FK constraint resolves
        # (matches the pattern in tests/orm/test_agentic_mode_default.py).
        conn.execute(
            text(
                "INSERT INTO thrive_user_role (id, role_name, description, role) "
                "VALUES (4, 'Patient', 'Patient access', 'PATIENT')"
            )
        )
        conn.execute(
            text(
                "INSERT INTO thrive_user "
                "(username, first_name, last_name, password, email, organization, user_role_id) "
                "VALUES ('legacy_thinking', 'Leg', 'Acy', 'x', 'legacy@example.com', 'Acme', 4)"
            )
        )

    # Apply the new migration.
    command.upgrade(cfg, "head")

    with engine.connect() as conn:
        val = conn.execute(
            text("SELECT show_thinking_process FROM thrive_user WHERE username='legacy_thinking'")
        ).scalar()
    assert val == 0, "existing user should be backfilled to show_thinking_process = False"
