"""agentic_mode should default ON for everybody — new users and existing.

- New users created via the ORM / create_user get agentic_mode = True.
- The data migration flips all pre-existing rows to True.
"""

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text

from orm.functions import create_user
from orm.models import User, UserRole

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_create_user_defaults_agentic_mode_on(in_memory_orm_session):
    with in_memory_orm_session() as session:
        role_id = session.query(UserRole).filter(UserRole.role_name == "Doctor").one().id

    assert (
        create_user(
            "newdoc",
            "pw",
            "New",
            "Doc",
            role_id,
            email="newdoc@example.com",
            organization="Acme",
        )
        is True
    )

    with in_memory_orm_session() as session:
        user = session.query(User).filter(User.username == "newdoc").one()
        assert user.agentic_mode is True


def test_migration_turns_on_agentic_mode_for_existing_users(tmp_path):
    """Upgrade to the prior revision, insert a user with agentic_mode=0,
    then upgrade to head — the new migration must flip it to 1."""
    url = f"sqlite:///{tmp_path / 'thrive.sqlite3'}"
    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", url)

    # Schema state just before the default-on migration.
    command.upgrade(cfg, "f3e688a55df6")

    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO thrive_user (username, first_name, last_name, password, agentic_mode) "
                "VALUES ('legacy', 'Leg', 'Acy', 'x', 0)"
            )
        )

    # Apply the default-on migration.
    command.upgrade(cfg, "head")

    with engine.connect() as conn:
        val = conn.execute(text("SELECT agentic_mode FROM thrive_user WHERE username='legacy'")).scalar()
    assert val == 1, "existing user should have been flipped to agentic_mode on"
