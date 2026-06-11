"""Required-field validation tests — Epic #179.

Focused on the new ``UserValidationError`` raised by ``create_user`` and
``update_user`` when ``email`` / ``organization`` / ``user_role_id`` are
missing, empty, malformed, or point at a non-existent role row.

Adjacent tests (``test_create_user_trim.py``, ``test_update_user_email_org.py``)
cover the happy paths and the per-field rejection cases that pre-date #179;
this file focuses on the *structure* of the error (``missing_fields``) and
the multi-field error case so the UI can rely on a stable contract when
rendering inline per-field messages.
"""

from __future__ import annotations

import pytest

from orm.functions import UserValidationError, create_user, update_user
from orm.models import User, UserRole


def _doctor_role_id(session_factory) -> int:
    with session_factory() as session:
        return session.query(UserRole).filter(UserRole.role_name == "Doctor").one().id


# ── create_user — single-field validation cases ──────────────────────────


def test_create_user_missing_email_raises_with_email_in_missing_fields(in_memory_orm_session):
    role_id = _doctor_role_id(in_memory_orm_session)
    with pytest.raises(UserValidationError) as exc:
        create_user("alice", "pw", "Alice", "Smith", role_id, email="", organization="Acme")
    assert exc.value.missing_fields == ["email"]


def test_create_user_missing_organization_raises_with_organization_in_missing_fields(in_memory_orm_session):
    role_id = _doctor_role_id(in_memory_orm_session)
    with pytest.raises(UserValidationError) as exc:
        create_user("alice", "pw", "Alice", "Smith", role_id, email="alice@example.com", organization="")
    assert exc.value.missing_fields == ["organization"]


def test_create_user_missing_role_raises_with_role_in_missing_fields(in_memory_orm_session):
    """``role_id`` is reported as ``"role"`` so the UI label matches."""
    with pytest.raises(UserValidationError) as exc:
        create_user("alice", "pw", "Alice", "Smith", None, email="alice@example.com", organization="Acme")
    assert exc.value.missing_fields == ["role"]


def test_create_user_zero_role_id_raises(in_memory_orm_session):
    """role_id=0 is not a positive int → role missing."""
    with pytest.raises(UserValidationError) as exc:
        create_user("alice", "pw", "Alice", "Smith", 0, email="alice@example.com", organization="Acme")
    assert "role" in exc.value.missing_fields


def test_create_user_nonexistent_role_id_raises(in_memory_orm_session):
    """A role_id that doesn't correspond to a thrive_user_role row → role missing."""
    with pytest.raises(UserValidationError) as exc:
        create_user(
            "alice",
            "pw",
            "Alice",
            "Smith",
            999_999,  # no such role
            email="alice@example.com",
            organization="Acme",
        )
    assert "role" in exc.value.missing_fields


# ── create_user — multi-field validation cases ───────────────────────────


def test_create_user_all_three_missing_lists_all_three_fields(in_memory_orm_session):
    """When all three required fields are missing, missing_fields lists them
    in a stable order (email, organization, role) so the UI can map each
    to the right input."""
    with pytest.raises(UserValidationError) as exc:
        create_user("alice", "pw", "Alice", "Smith", None, email="", organization="")
    assert exc.value.missing_fields == ["email", "organization", "role"]


def test_create_user_email_and_organization_missing(in_memory_orm_session):
    role_id = _doctor_role_id(in_memory_orm_session)
    with pytest.raises(UserValidationError) as exc:
        create_user("alice", "pw", "Alice", "Smith", role_id, email="bad", organization="")
    assert exc.value.missing_fields == ["email", "organization"]


def test_create_user_validation_runs_before_db_write(in_memory_orm_session):
    """A validation failure leaves no rows behind even when other fields are valid."""
    role_id = _doctor_role_id(in_memory_orm_session)
    with pytest.raises(UserValidationError):
        create_user("alice", "pw", "Alice", "Smith", role_id, email="not-an-email", organization="Acme")

    with in_memory_orm_session() as session:
        assert session.query(User).filter(User.username == "alice").count() == 0


# ── UserValidationError shape ────────────────────────────────────────────


def test_user_validation_error_carries_missing_fields_list():
    """The exception exposes a typed ``missing_fields`` list — that contract
    is what the dialogs depend on for per-field message rendering."""
    err = UserValidationError(["email", "role"])
    assert err.missing_fields == ["email", "role"]
    assert "email" in str(err)
    assert "role" in str(err)


def test_user_validation_error_is_value_error_subclass():
    """``UserValidationError`` MUST subclass ``ValueError`` so legacy
    callers that catch ``ValueError`` still trip (compatibility)."""
    assert issubclass(UserValidationError, ValueError)


# ── update_user — per-field validation ───────────────────────────────────


def test_update_user_passing_empty_organization_raises(in_memory_orm_session):
    role_id = _doctor_role_id(in_memory_orm_session)
    create_user("alice", "pw", "Alice", "Smith", role_id, email="alice@example.com", organization="Acme")

    with in_memory_orm_session() as session:
        user_id = session.query(User).filter(User.username == "alice").one().id

    with pytest.raises(UserValidationError) as exc:
        update_user(user_id, organization="")
    assert "organization" in exc.value.missing_fields


def test_update_user_with_invalid_role_id_raises(in_memory_orm_session):
    role_id = _doctor_role_id(in_memory_orm_session)
    create_user("alice", "pw", "Alice", "Smith", role_id, email="alice@example.com", organization="Acme")

    with in_memory_orm_session() as session:
        user_id = session.query(User).filter(User.username == "alice").one().id

    with pytest.raises(UserValidationError) as exc:
        update_user(user_id, role_id=999_999)
    assert "role" in exc.value.missing_fields


def test_update_user_omitted_fields_are_not_validated(in_memory_orm_session):
    """``None`` means "don't touch" — omitting email/org/role from the
    kwargs should not trigger validation. (Legacy semantics preserved.)"""
    role_id = _doctor_role_id(in_memory_orm_session)
    create_user("alice", "pw", "Alice", "Smith", role_id, email="alice@example.com", organization="Acme")

    with in_memory_orm_session() as session:
        user_id = session.query(User).filter(User.username == "alice").one().id

    # Touch only first_name — no validation error should be raised.
    assert update_user(user_id, first_name="Alyce") is True
