"""create_user must trim whitespace from username/name fields.

A leading/trailing space in the username (e.g. from the admin create-user
form) silently breaks login, because verify_user_credentials matches on the
exact stored username. Regression test for the `" adlouhy"` case seen in dev.
"""

from orm.functions import create_user
from orm.models import User, UserRole


def _doctor_role_id(session_factory):
    with session_factory() as session:
        return session.query(UserRole).filter(UserRole.role_name == "Doctor").one().id


def test_create_user_trims_username_and_names(in_memory_orm_session):
    role_id = _doctor_role_id(in_memory_orm_session)

    ok = create_user(
        username="  adlouhy  ",
        password="pw",
        first_name="  Adrienne ",
        last_name=" Dlouhy  ",
        role_id=role_id,
    )
    assert ok is True

    with in_memory_orm_session() as session:
        user = session.query(User).filter(User.username == "adlouhy").one_or_none()
        assert user is not None, "username should be stored trimmed"
        assert user.first_name == "Adrienne"
        assert user.last_name == "Dlouhy"


def test_create_user_dedupes_against_trimmed_username(in_memory_orm_session):
    """A padded duplicate of an existing username must be rejected."""
    role_id = _doctor_role_id(in_memory_orm_session)

    assert create_user("adlouhy", "pw", "A", "D", role_id) is True
    # Same username with surrounding whitespace should be treated as a dup.
    assert create_user("  adlouhy  ", "pw", "A", "D", role_id) is False
