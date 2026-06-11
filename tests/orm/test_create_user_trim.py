"""create_user behavior tests — whitespace trimming, required email + organization,
email format validation, email uniqueness (case-insensitive), and organization
stripping.

Original purpose: regression test for the `" adlouhy"` whitespace-in-username
case that silently broke login. Extended in Epic #98 / Feature Spec #99 to
cover the new required `email` and `organization` keyword-only parameters
and their validation gates.
"""

import pytest

from orm.functions import UserValidationError, create_user
from orm.models import User, UserRole

# Default keyword args every test that doesn't care about email / org passes
# so existing tests keep focusing on what they actually exercise.
_VALID = dict(email="user@example.com", organization="Acme")


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
        **_VALID,
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

    assert create_user("adlouhy", "pw", "A", "D", role_id, **_VALID) is True
    # Same username with surrounding whitespace should be treated as a dup.
    # Note: still pass valid email/org so the rejection is definitively for
    # the username collision, not for missing fields.
    assert (
        create_user(
            "  adlouhy  ",
            "pw",
            "A",
            "D",
            role_id,
            email="different@example.com",
            organization="Acme",
        )
        is False
    )


# ── New tests for email + organization (Epic #98 / Feature Spec #99) ──────


def test_email_and_organization_persist_after_create(in_memory_orm_session):
    role_id = _doctor_role_id(in_memory_orm_session)

    ok = create_user(
        "alice",
        "pw",
        "Alice",
        "Smith",
        role_id,
        email="alice@example.com",
        organization="Acme",
    )
    assert ok is True

    with in_memory_orm_session() as session:
        user = session.query(User).filter(User.username == "alice").one()
        assert user.email == "alice@example.com"
        assert user.organization == "Acme"


@pytest.mark.parametrize(
    "bad_email",
    ["foo", "@bar.com", "foo@", "foo@bar", "", "   "],
)
def test_email_format_invalid_rejected(in_memory_orm_session, bad_email):
    """Epic #179: bad emails raise UserValidationError with `email` in missing_fields."""
    role_id = _doctor_role_id(in_memory_orm_session)
    with pytest.raises(UserValidationError) as exc_info:
        create_user(
            "alice",
            "pw",
            "Alice",
            "Smith",
            role_id,
            email=bad_email,
            organization="Acme",
        )
    assert "email" in exc_info.value.missing_fields

    with in_memory_orm_session() as session:
        assert session.query(User).filter(User.username == "alice").first() is None


def test_email_duplicate_rejected_case_insensitive(in_memory_orm_session):
    role_id = _doctor_role_id(in_memory_orm_session)

    assert (
        create_user(
            "alice",
            "pw",
            "Alice",
            "Smith",
            role_id,
            email="User@Example.com",
            organization="Acme",
        )
        is True
    )
    # Different username but same email (different case) — must be rejected.
    assert (
        create_user(
            "bob",
            "pw",
            "Bob",
            "Jones",
            role_id,
            email="user@example.com",
            organization="Acme",
        )
        is False
    )

    with in_memory_orm_session() as session:
        assert session.query(User).filter(User.username == "bob").first() is None


def test_organization_empty_after_strip_rejected(in_memory_orm_session):
    """Epic #179: whitespace-only org raises UserValidationError."""
    role_id = _doctor_role_id(in_memory_orm_session)
    with pytest.raises(UserValidationError) as exc_info:
        create_user(
            "alice",
            "pw",
            "Alice",
            "Smith",
            role_id,
            email="alice@example.com",
            organization="   ",
        )
    assert "organization" in exc_info.value.missing_fields

    with in_memory_orm_session() as session:
        assert session.query(User).filter(User.username == "alice").first() is None


def test_email_and_organization_stripped_before_persist(in_memory_orm_session):
    role_id = _doctor_role_id(in_memory_orm_session)

    ok = create_user(
        "alice",
        "pw",
        "Alice",
        "Smith",
        role_id,
        email="  alice@example.com  ",
        organization="  Acme  ",
    )
    assert ok is True

    with in_memory_orm_session() as session:
        user = session.query(User).filter(User.username == "alice").one()
        assert user.email == "alice@example.com"
        assert user.organization == "Acme"
