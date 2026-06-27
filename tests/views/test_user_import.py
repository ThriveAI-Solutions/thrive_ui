"""End-to-end smoke test for views._admin_helpers.import_users().

Drives the importer against an in-memory SQLite using a constructed pandas
DataFrame (no file I/O) and asserts the full mixed-row behavior: valid rows
persist with email + organization populated; rows missing either field are
skipped; malformed-email / duplicate-username / duplicate-email rows are
skipped via the create_user() delegation introduced by Feature Spec #100.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd

from orm.functions import create_user
from orm.models import User, UserRole


def _doctor_role_id(session_factory) -> int:
    with session_factory() as session:
        return session.query(UserRole).filter(UserRole.role_name == "Doctor").one().id


def _df_with_mixed_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # Valid row 1
            {
                "UserID": "alice",
                "start_password": "pw",
                "First Name ": "Alice",
                "Last Name ": "Smith",
                "Email": "alice@example.com",
                "Organization": "Acme",
            },
            # Valid row 2
            {
                "UserID": "bob",
                "start_password": "pw",
                "First Name ": "Bob",
                "Last Name ": "Jones",
                "Email": "bob@example.com",
                "Organization": "Acme",
            },
            # Missing email
            {
                "UserID": "charlie",
                "start_password": "pw",
                "First Name ": "Charlie",
                "Last Name ": "Brown",
                "Email": float("nan"),
                "Organization": "Acme",
            },
            # Missing organization
            {
                "UserID": "dave",
                "start_password": "pw",
                "First Name ": "Dave",
                "Last Name ": "Lee",
                "Email": "dave@example.com",
                "Organization": float("nan"),
            },
            # Malformed email
            {
                "UserID": "eve",
                "start_password": "pw",
                "First Name ": "Eve",
                "Last Name ": "Khan",
                "Email": "not-an-email",
                "Organization": "Acme",
            },
            # Duplicate username (pre-seeded as "dup-username" below)
            {
                "UserID": "dup-username",
                "start_password": "pw",
                "First Name ": "Dup",
                "Last Name ": "User",
                "Email": "different@example.com",
                "Organization": "Acme",
            },
            # Duplicate email (case-different from pre-seeded "existing@example.com")
            {
                "UserID": "frank",
                "start_password": "pw",
                "First Name ": "Frank",
                "Last Name ": "Adams",
                "Email": "EXISTING@example.com",
                "Organization": "Acme",
            },
        ]
    )


def test_import_users_handles_mixed_rows(in_memory_orm_session, monkeypatch):
    """7-row import: 2 valid + 5 failures (missing email, missing org, bad
    format, dup username, dup email). Asserts the persisted state and the
    Streamlit error/success surface."""
    # views/_admin_helpers.py does `from orm.models import SessionLocal, ...` which
    # re-binds the symbol into views._admin_helpers's namespace. The conftest fixture
    # only patches orm.models.SessionLocal and orm.functions.SessionLocal —
    # we need to additionally patch views._admin_helpers.SessionLocal so the importer's
    # role-lookup block uses the in-memory engine.
    import orm.models  # noqa: F401 — ensure module is loaded so we can introspect SessionLocal
    import views._admin_helpers

    monkeypatch.setattr(views._admin_helpers, "SessionLocal", views._admin_helpers.SessionLocal)
    # The fixture has already replaced orm.models.SessionLocal with the
    # in-memory sessionmaker, but views._admin_helpers's local binding pre-dates that
    # patch — re-import it now.
    from orm.models import SessionLocal as _new_session_local

    monkeypatch.setattr(views._admin_helpers, "SessionLocal", _new_session_local)

    role_id = _doctor_role_id(in_memory_orm_session)

    # orm.functions._get_current_user_id reads st.session_state.cookies; outside
    # a Streamlit runtime that raises AttributeError. Patch orm.functions.st
    # with a MagicMock so the admin-action audit-log path no-ops cleanly.
    import orm.functions

    mock_orm_st = MagicMock()
    mock_orm_st.session_state.cookies.get.return_value = None
    monkeypatch.setattr(orm.functions, "st", mock_orm_st)

    # Pre-seed one user for duplicate detection (username = "dup-username",
    # email = "existing@example.com").
    assert (
        create_user(
            "dup-username",
            "pw",
            "Dup",
            "Existing",
            role_id,
            email="existing@example.com",
            organization="Acme",
        )
        is True
    )

    # Build the DataFrame and patch the file-discovery / Excel-load path so
    # import_users() doesn't touch the disk.
    df = _df_with_mixed_rows()

    mock_st = MagicMock()
    # st.expander is used as a context manager; make it return a MagicMock.
    mock_st.expander.return_value.__enter__ = lambda self_: self_
    mock_st.expander.return_value.__exit__ = lambda self_, *args: False

    with (
        patch.object(views._admin_helpers, "get_user_list_excel", return_value=df),
        patch.object(views._admin_helpers, "st", mock_st),
        patch("os.path.exists", return_value=True),
    ):
        result = views._admin_helpers.import_users()

    # import_users returns True when any user was successfully imported.
    assert result is True

    # Persisted state: pre-seeded user plus exactly 2 new rows (alice, bob).
    with in_memory_orm_session() as session:
        rows = session.query(User).order_by(User.username).all()
        usernames = [u.username for u in rows]
        assert usernames == ["alice", "bob", "dup-username"]

        alice = next(u for u in rows if u.username == "alice")
        bob = next(u for u in rows if u.username == "bob")
        assert alice.email == "alice@example.com"
        assert alice.organization == "Acme"
        assert bob.email == "bob@example.com"
        assert bob.organization == "Acme"

    # Success surface: success was called with a message that includes "2".
    assert mock_st.success.called, "st.success was not called"
    success_args = " ".join(str(a) for call in mock_st.success.call_args_list for a in call.args)
    assert "2" in success_args

    # Failure surface: all 5 expected failures appear in st.text calls inside
    # the failures expander.
    text_calls = " ".join(str(a) for call in mock_st.text.call_args_list for a in call.args)
    # Charlie: "Missing email"
    assert "charlie" in text_calls.lower() or "Missing email" in text_calls
    # Dave: "Missing organization"
    assert "dave" in text_calls.lower() or "Missing organization" in text_calls
    # Eve: malformed email -> create_user rejected
    assert "eve" in text_calls.lower()
    # dup-username: duplicate username -> create_user rejected
    assert "dup-username" in text_calls
    # Frank: duplicate email -> create_user rejected
    assert "frank" in text_calls.lower()
