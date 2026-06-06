"""Tests for update_user() — email + organization keyword params.

Feature Spec #101: extends `update_user()` with optional `email` and
`organization` keyword parameters that share the validation gate
`create_user()` already uses (lightweight regex + case-insensitive
uniqueness for email, non-empty-after-strip for organization). The
existing optional-update semantics are preserved — `None` means "don't
touch."
"""

from __future__ import annotations

import pytest

from orm.functions import create_user, update_user
from orm.models import User, UserRole


def _doctor_role_id(session_factory) -> int:
    with session_factory() as session:
        return session.query(UserRole).filter(UserRole.role_name == "Doctor").one().id


def _seed_alice(session_factory, role_id: int, email: str = "alice@example.com") -> int:
    """Seed Alice via create_user; return her user_id."""
    assert (
        create_user(
            "alice",
            "pw",
            "Alice",
            "Smith",
            role_id,
            email=email,
            organization="Acme",
        )
        is True
    )
    with session_factory() as session:
        return session.query(User).filter(User.username == "alice").one().id


# ── Happy path + field setting ────────────────────────────────────────────


def test_update_user_sets_email_and_organization(in_memory_orm_session):
    role_id = _doctor_role_id(in_memory_orm_session)
    user_id = _seed_alice(in_memory_orm_session, role_id)

    ok = update_user(
        user_id,
        email="alice.new@example.com",
        organization="Globex",
    )
    assert ok is True

    with in_memory_orm_session() as session:
        user = session.query(User).filter(User.id == user_id).one()
        assert user.email == "alice.new@example.com"
        assert user.organization == "Globex"


# ── Validation gates ──────────────────────────────────────────────────────


@pytest.mark.parametrize("bad_email", ["foo", "foo@", "@bar.com", "foo@bar"])
def test_update_user_rejects_invalid_email_format(in_memory_orm_session, bad_email):
    role_id = _doctor_role_id(in_memory_orm_session)
    user_id = _seed_alice(in_memory_orm_session, role_id)

    ok = update_user(user_id, email=bad_email)
    assert ok is False

    with in_memory_orm_session() as session:
        user = session.query(User).filter(User.id == user_id).one()
        # Email unchanged on rejection.
        assert user.email == "alice@example.com"


def test_update_user_rejects_duplicate_email_case_insensitive(in_memory_orm_session):
    role_id = _doctor_role_id(in_memory_orm_session)
    alice_id = _seed_alice(in_memory_orm_session, role_id, email="alice@example.com")

    # Seed a second user that owns the email we'll try to steal.
    assert (
        create_user(
            "bob",
            "pw",
            "Bob",
            "Jones",
            role_id,
            email="Other@Example.com",
            organization="Acme",
        )
        is True
    )

    # Try to update Alice's email to a case-different duplicate of Bob's.
    ok = update_user(alice_id, email="other@example.com")
    assert ok is False

    with in_memory_orm_session() as session:
        user = session.query(User).filter(User.id == alice_id).one()
        assert user.email == "alice@example.com"


def test_update_user_rejects_empty_organization_after_strip(in_memory_orm_session):
    role_id = _doctor_role_id(in_memory_orm_session)
    user_id = _seed_alice(in_memory_orm_session, role_id)

    ok = update_user(user_id, organization="   ")
    assert ok is False

    with in_memory_orm_session() as session:
        user = session.query(User).filter(User.id == user_id).one()
        assert user.organization == "Acme"


# ── Optional-update semantics preserved ───────────────────────────────────


def test_update_user_leaves_email_unchanged_when_kwarg_omitted(in_memory_orm_session):
    """Existing semantics: only fields with non-None kwargs are updated."""
    role_id = _doctor_role_id(in_memory_orm_session)
    user_id = _seed_alice(in_memory_orm_session, role_id)

    # Touch only the first_name; email + organization must survive.
    ok = update_user(user_id, first_name="Alyce")
    assert ok is True

    with in_memory_orm_session() as session:
        user = session.query(User).filter(User.id == user_id).one()
        assert user.first_name == "Alyce"
        assert user.email == "alice@example.com"
        assert user.organization == "Acme"


# ── Whitespace handling ───────────────────────────────────────────────────


def test_update_user_strips_whitespace_from_email_and_organization(in_memory_orm_session):
    role_id = _doctor_role_id(in_memory_orm_session)
    user_id = _seed_alice(in_memory_orm_session, role_id)

    ok = update_user(
        user_id,
        email="  alice.new@example.com  ",
        organization="  Globex  ",
    )
    assert ok is True

    with in_memory_orm_session() as session:
        user = session.query(User).filter(User.id == user_id).one()
        assert user.email == "alice.new@example.com"
        assert user.organization == "Globex"
