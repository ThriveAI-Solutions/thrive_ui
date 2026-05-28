"""Stale/invalid cookie handling for local auth.

Regression tests for the cascade where a cookie holds a ``user_id`` that no
longer exists in the DB (e.g. after the SQLite DB was recreated/migrated). The
old behavior dereferenced a ``None`` user three layers deep, producing:
    - "Error setting user preferences in session state: 'NoneType'... 'show_sql'"
    - "Error checking authentication: 'NoneType'... 'first_name'"
    - "st.session_state has no attribute 'username'"

Desired behavior: treat the cookie as invalid, clear it, and show the login
form — no error spew.
"""

from datetime import datetime, timedelta
from unittest.mock import patch


class FakeSessionState(dict):
    """Dict that also supports attribute access, like st.session_state."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


class FakeCookies:
    def __init__(self, initial=None):
        self._d = dict(initial or {})
        self.saved = False

    def get(self, key, default=None):
        return self._d.get(key, default)

    def __contains__(self, key):
        return key in self._d

    def __setitem__(self, key, value):
        self._d[key] = value

    def __getitem__(self, key):
        return self._d[key]

    def save(self):
        self.saved = True


def test_set_preferences_returns_none_without_error_when_user_missing():
    """get_user() -> None must short-circuit, not crash on user.show_sql."""
    import orm.functions as functions

    session_state = FakeSessionState()
    session_state.cookies = FakeCookies({"user_id": "99"})

    with (
        patch("streamlit.session_state", session_state),
        patch.object(functions, "get_user", return_value=None),
        patch("streamlit.error") as error_mock,
    ):
        result = functions.set_user_preferences_in_session_state()

    assert result is None
    error_mock.assert_not_called()
    assert "username" not in session_state
    assert "show_sql" not in session_state


def test_handle_local_auth_clears_stale_cookie_and_shows_login():
    """An unresolvable user_id cookie => clear cookie + render login form."""
    import utils.auth as auth_module

    expiry = (datetime.now() + timedelta(hours=8)).isoformat()
    session_state = FakeSessionState()
    session_state.cookies = FakeCookies({"user_id": "99", "expiry_date": expiry})

    with (
        patch("streamlit.session_state", session_state),
        patch.object(auth_module, "set_user_preferences_in_session_state", return_value=None),
        patch.object(auth_module, "_show_local_login") as login_mock,
        patch("streamlit.error") as error_mock,
    ):
        auth_module._handle_local_auth()

    login_mock.assert_called_once()
    error_mock.assert_not_called()
    assert session_state.cookies.get("user_id") == ""
    assert session_state.cookies.saved is True
