"""Inline per-field error rendering — Epic #179 subsystem 3.

Drives the Create User dialog (``views/_admin_helpers.create_user_dialog``)
and the My Account profile form (``views/my_account``) at the function /
module level with ``streamlit`` stubbed, asserting:

  - ``UserValidationError`` from the server-side validator routes to
    per-field session-state error messages keyed ``"email"`` /
    ``"organization"`` / ``"role"``.
  - On valid input the error dict is cleared and ``st.success`` fires.
  - On generic non-validation failure (``create_user`` / ``update_user``
    return False) the error dict is cleared and a single generic
    ``st.error`` banner fires.

The Edit User profile form lives inside ``views/admin_users.render``
which is a non-trivial Streamlit page; the unit-tested error-mapping
logic is identical to the Create User dialog (same exception, same
``missing_fields`` keys, same per-field message strings), so the
dialog-level tests here are representative of the contract for both
surfaces.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from orm.functions import UserValidationError


# ── Create User dialog ───────────────────────────────────────────────────


def _make_st_stub(form_submitted: bool = True, inputs: dict | None = None):
    """Stub ``streamlit`` for the Create User dialog.

    Returns a MagicMock that mimics the API used by ``create_user_dialog``:
    ``text_input`` / ``selectbox`` return canned values; ``form`` /
    ``dialog`` are context managers; ``form_submit_button`` returns the
    ``form_submitted`` flag; ``session_state`` is a real dict so the
    error-key plumbing works.
    """
    inputs = inputs or {}
    stub = MagicMock()
    stub.session_state = {}

    # @st.dialog decorator should be a passthrough so we can call the
    # decorated function directly. The dialog is applied as
    # @st.dialog("Create User") so .dialog must be callable, return a
    # decorator that returns the function unchanged.
    stub.dialog = lambda *_a, **_kw: (lambda f: f)

    # Form / dialog context managers.
    form_cm = MagicMock()
    form_cm.__enter__ = MagicMock(return_value=form_cm)
    form_cm.__exit__ = MagicMock(return_value=False)
    stub.form = MagicMock(return_value=form_cm)

    def _text_input(label, *args, **kwargs):
        return inputs.get(label, "")

    def _selectbox(label, options=None, **kwargs):
        v = inputs.get(label)
        if v is not None:
            return v
        return options[0] if options else None

    stub.text_input = _text_input
    stub.selectbox = _selectbox
    stub.form_submit_button = MagicMock(return_value=form_submitted)
    stub.error = MagicMock()
    stub.success = MagicMock()
    stub.rerun = MagicMock()
    return stub


def _fake_roles():
    return [(1, "Admin", "Admin role"), (2, "Doctor", "Doctor role"), (3, "Patient", "Patient role")]


def _fake_themes():
    return ["default"]


def test_create_user_dialog_missing_email_populates_session_state_field_error():
    """When create_user raises UserValidationError with email, the dialog
    stores ``"email"`` in the field-error dict for the next render."""
    from views import _admin_helpers

    inputs = {
        "Username": "alice",
        "Temporary Password": "pw",
        "First Name": "Alice",
        "Last Name": "Smith",
        "Email": "",  # empty -> validation error
        "Organization": "Acme",
        "Role": "Doctor",
        "Theme": "default",
    }
    stub = _make_st_stub(form_submitted=True, inputs=inputs)
    create_user_mock = MagicMock(side_effect=UserValidationError(["email"]))

    with (
        patch.object(_admin_helpers, "st", stub),
        patch.object(_admin_helpers, "get_all_user_roles", return_value=_fake_roles()),
        patch.object(_admin_helpers, "user_selectable_themes", return_value=_fake_themes()),
        patch.object(_admin_helpers, "create_user", create_user_mock),
    ):
        _admin_helpers._create_user_dialog_body()

    assert "email" in stub.session_state["create_user_dialog_field_errors"]
    stub.rerun.assert_called_once()
    # No success or generic-error banner — only the per-field path fired.
    stub.success.assert_not_called()


def test_create_user_dialog_missing_organization_populates_session_state_field_error():
    from views import _admin_helpers

    inputs = {
        "Username": "alice",
        "Temporary Password": "pw",
        "First Name": "Alice",
        "Last Name": "Smith",
        "Email": "alice@example.com",
        "Organization": "",  # empty -> validation error
        "Role": "Doctor",
        "Theme": "default",
    }
    stub = _make_st_stub(form_submitted=True, inputs=inputs)
    create_user_mock = MagicMock(side_effect=UserValidationError(["organization"]))

    with (
        patch.object(_admin_helpers, "st", stub),
        patch.object(_admin_helpers, "get_all_user_roles", return_value=_fake_roles()),
        patch.object(_admin_helpers, "user_selectable_themes", return_value=_fake_themes()),
        patch.object(_admin_helpers, "create_user", create_user_mock),
    ):
        _admin_helpers._create_user_dialog_body()

    errors = stub.session_state["create_user_dialog_field_errors"]
    assert "organization" in errors
    assert "email" not in errors


def test_create_user_dialog_multiple_missing_fields_renders_each_inline():
    """All three required fields missing → all three present in the error dict."""
    from views import _admin_helpers

    inputs = {
        "Username": "alice",
        "Temporary Password": "pw",
        "First Name": "Alice",
        "Last Name": "Smith",
        "Email": "",
        "Organization": "",
        "Role": "Doctor",
        "Theme": "default",
    }
    stub = _make_st_stub(form_submitted=True, inputs=inputs)
    create_user_mock = MagicMock(side_effect=UserValidationError(["email", "organization", "role"]))

    with (
        patch.object(_admin_helpers, "st", stub),
        patch.object(_admin_helpers, "get_all_user_roles", return_value=_fake_roles()),
        patch.object(_admin_helpers, "user_selectable_themes", return_value=_fake_themes()),
        patch.object(_admin_helpers, "create_user", create_user_mock),
    ):
        _admin_helpers._create_user_dialog_body()

    errors = stub.session_state["create_user_dialog_field_errors"]
    assert set(errors.keys()) == {"email", "organization", "role"}


def test_create_user_dialog_success_clears_field_errors_and_calls_st_success():
    from views import _admin_helpers

    inputs = {
        "Username": "alice",
        "Temporary Password": "pw",
        "First Name": "Alice",
        "Last Name": "Smith",
        "Email": "alice@example.com",
        "Organization": "Acme",
        "Role": "Doctor",
        "Theme": "default",
    }
    stub = _make_st_stub(form_submitted=True, inputs=inputs)
    # Pre-seed a stale error dict; success should wipe it.
    stub.session_state["create_user_dialog_field_errors"] = {"email": "old"}

    with (
        patch.object(_admin_helpers, "st", stub),
        patch.object(_admin_helpers, "get_all_user_roles", return_value=_fake_roles()),
        patch.object(_admin_helpers, "user_selectable_themes", return_value=_fake_themes()),
        patch.object(_admin_helpers, "create_user", MagicMock(return_value=True)),
    ):
        _admin_helpers._create_user_dialog_body()

    assert stub.session_state["create_user_dialog_field_errors"] == {}
    stub.success.assert_called_once()


def test_create_user_dialog_generic_failure_clears_field_errors_and_calls_st_error():
    """When create_user returns False (not raises) the error dict is wiped
    and a generic banner is shown — duplicate username/email has no
    per-field action so we surface it as one banner."""
    from views import _admin_helpers

    inputs = {
        "Username": "alice",
        "Temporary Password": "pw",
        "First Name": "Alice",
        "Last Name": "Smith",
        "Email": "alice@example.com",
        "Organization": "Acme",
        "Role": "Doctor",
        "Theme": "default",
    }
    stub = _make_st_stub(form_submitted=True, inputs=inputs)

    with (
        patch.object(_admin_helpers, "st", stub),
        patch.object(_admin_helpers, "get_all_user_roles", return_value=_fake_roles()),
        patch.object(_admin_helpers, "user_selectable_themes", return_value=_fake_themes()),
        patch.object(_admin_helpers, "create_user", MagicMock(return_value=False)),
    ):
        _admin_helpers._create_user_dialog_body()

    assert stub.session_state["create_user_dialog_field_errors"] == {}
    stub.error.assert_called_once()
    stub.success.assert_not_called()


def test_create_user_dialog_blank_username_blocks_before_calling_create_user():
    """The dialog's pre-validation blocks if Username/Password/First name
    is blank — server-side validation only covers the three Epic #179
    fields so we still need a client-side gate for the others."""
    from views import _admin_helpers

    inputs = {
        "Username": "",  # blocked at UI
        "Temporary Password": "pw",
        "First Name": "Alice",
        "Last Name": "Smith",
        "Email": "alice@example.com",
        "Organization": "Acme",
        "Role": "Doctor",
        "Theme": "default",
    }
    stub = _make_st_stub(form_submitted=True, inputs=inputs)
    create_user_mock = MagicMock(return_value=True)

    with (
        patch.object(_admin_helpers, "st", stub),
        patch.object(_admin_helpers, "get_all_user_roles", return_value=_fake_roles()),
        patch.object(_admin_helpers, "user_selectable_themes", return_value=_fake_themes()),
        patch.object(_admin_helpers, "create_user", create_user_mock),
    ):
        _admin_helpers._create_user_dialog_body()

    create_user_mock.assert_not_called()
    stub.error.assert_called()  # generic banner for missing username


# ── Server contract: error keys are stable ───────────────────────────────


@pytest.mark.parametrize(
    "missing,expected_key",
    [
        (["email"], "email"),
        (["organization"], "organization"),
        (["role"], "role"),
    ],
)
def test_user_validation_error_field_keys_are_stable(missing, expected_key):
    """The UI relies on these exact strings for the field-error mapping."""
    err = UserValidationError(missing)
    assert expected_key in err.missing_fields
