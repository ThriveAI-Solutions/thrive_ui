"""Tests for orm.functions.get_users_for_export (#122)."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from orm.models import AdminAction, AdminActionType, Base, RoleTypeEnum, User, UserRole


@pytest.fixture
def session_factory(monkeypatch):
    """In-memory SQLite with SessionLocal patched on both functions and logging_functions."""
    from orm import functions as fns
    from orm import logging_functions as lf

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    monkeypatch.setattr(fns, "SessionLocal", Session)
    monkeypatch.setattr(lf, "SessionLocal", Session)
    return Session


def _seed_roles(session):
    session.add_all(
        [
            UserRole(id=1, role_name="Admin", description="Admin", role=RoleTypeEnum.ADMIN),
            UserRole(id=2, role_name="Doctor", description="Doctor", role=RoleTypeEnum.DOCTOR),
            UserRole(id=3, role_name="Nurse", description="Nurse", role=RoleTypeEnum.NURSE),
            UserRole(id=4, role_name="Patient", description="Patient", role=RoleTypeEnum.PATIENT),
        ]
    )
    session.commit()


def _seed_user(session, *, id, username, first, last, email, org, role_id, password="hash:abc"):
    session.add(
        User(
            id=id,
            user_role_id=role_id,
            username=username,
            first_name=first,
            last_name=last,
            password=password,
            email=email,
            organization=org,
        )
    )
    session.commit()


def test_admin_action_type_includes_user_export():
    assert AdminActionType.USER_EXPORT.value == "user_export"


def test_admin_caller_returns_rows_and_writes_audit(session_factory):
    from orm.functions import get_users_for_export

    s = session_factory()
    _seed_roles(s)
    _seed_user(s, id=1, username="admin1", first="Ada", last="Min", email="a@x.io", org="Org1", role_id=1)
    _seed_user(s, id=2, username="doc1", first="Don", last="Doc", email="d@x.io", org="Org2", role_id=2)
    _seed_user(s, id=3, username="pat1", first="Pat", last="Ient", email="p@x.io", org="Org3", role_id=4)

    rows, n = get_users_for_export(admin_id=1)

    assert n == 3
    assert len(rows) == 3
    # Audit row written
    audit = s.query(AdminAction).all()
    assert len(audit) == 1
    assert audit[0].admin_id == 1
    assert audit[0].action_type == "user_export"
    assert audit[0].affected_count == 3
    assert audit[0].success is True


def test_admin_caller_rows_have_exact_column_keys_in_order(session_factory):
    from orm.functions import get_users_for_export

    s = session_factory()
    _seed_roles(s)
    _seed_user(s, id=1, username="admin1", first="Ada", last="Min", email="a@x.io", org="Org1", role_id=1)

    rows, _ = get_users_for_export(admin_id=1)

    expected_keys = ["UserID", "First Name", "Last Name", "Email", "Organization", "Role"]
    assert list(rows[0].keys()) == expected_keys


def test_admin_caller_row_values_match_user_fields(session_factory):
    from orm.functions import get_users_for_export

    s = session_factory()
    _seed_roles(s)
    _seed_user(s, id=1, username="admin1", first="Ada", last="Min", email="a@x.io", org="Org1", role_id=1)

    rows, _ = get_users_for_export(admin_id=1)
    row = rows[0]

    assert row["UserID"] == "admin1"
    assert row["First Name"] == "Ada"
    assert row["Last Name"] == "Min"
    assert row["Email"] == "a@x.io"
    assert row["Organization"] == "Org1"
    assert row["Role"] == "Admin"


def test_returned_rows_exclude_sensitive_and_preference_fields(session_factory):
    from orm.functions import get_users_for_export

    s = session_factory()
    _seed_roles(s)
    _seed_user(s, id=1, username="admin1", first="Ada", last="Min", email="a@x.io", org="Org1", role_id=1)

    rows, _ = get_users_for_export(admin_id=1)
    row = rows[0]

    forbidden = {
        "password",
        "okta_sub",
        "theme",
        "show_sql",
        "show_table",
        "agentic_mode",
        "selected_llm_model",
        "selected_llm_provider",
        "show_chart",
        "voice_input",
    }
    assert not (set(row.keys()) & forbidden)


@pytest.mark.parametrize(
    "non_admin_role_id",
    [2, 3, 4],
    ids=["doctor", "nurse", "patient"],
)
def test_non_admin_caller_rejects_and_logs_failure(session_factory, non_admin_role_id):
    from orm.functions import get_users_for_export

    s = session_factory()
    _seed_roles(s)
    _seed_user(s, id=1, username="admin1", first="A", last="M", email="a@x.io", org="O", role_id=1)
    _seed_user(s, id=99, username="someone", first="S", last="O", email="s@x.io", org="O", role_id=non_admin_role_id)

    rows, n = get_users_for_export(admin_id=99)

    assert rows == []
    assert n == 0
    audit = s.query(AdminAction).all()
    assert len(audit) == 1
    assert audit[0].admin_id == 99
    assert audit[0].action_type == "user_export"
    assert audit[0].success is False
    assert audit[0].error_message  # non-empty rejection message


def test_unknown_caller_rejects_and_logs_failure(session_factory):
    from orm.functions import get_users_for_export

    s = session_factory()
    _seed_roles(s)
    # No user with id=999 exists.

    rows, n = get_users_for_export(admin_id=999)

    assert (rows, n) == ([], 0)
    audit = s.query(AdminAction).all()
    assert len(audit) == 1
    assert audit[0].success is False


def test_admin_only_roster_succeeds_with_one_row(session_factory):
    """Smallest legitimate success case: the admin themselves is the only user."""
    from orm.functions import get_users_for_export

    s = session_factory()
    _seed_roles(s)
    _seed_user(s, id=1, username="admin1", first="A", last="M", email="a@x.io", org="O", role_id=1)

    rows, n = get_users_for_export(admin_id=1)

    assert n == 1
    assert rows[0]["UserID"] == "admin1"
    audit = s.query(AdminAction).all()
    assert len(audit) == 1
    assert audit[0].success is True
    assert audit[0].affected_count == 1
