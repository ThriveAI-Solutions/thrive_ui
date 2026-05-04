"""Verifies utils.auth.check_authenticate dispatches between OIDC and local."""

from unittest.mock import patch


def test_check_authenticate_routes_to_local_when_auth_section_absent():
    """Default behavior unchanged: no [auth] section → existing local flow runs."""
    import utils.auth as auth_module
    import utils.okta_auth as okta_module

    with (
        patch("streamlit.secrets", new={}),
        patch.object(auth_module, "_handle_local_auth") as local_mock,
        patch.object(okta_module, "handle_oidc_auth") as oidc_mock,
    ):
        auth_module.check_authenticate()

    local_mock.assert_called_once()
    oidc_mock.assert_not_called()


def test_check_authenticate_routes_to_oidc_when_mode_is_oidc():
    """[auth].mode = 'oidc' routes to handle_oidc_auth."""
    import utils.auth as auth_module
    import utils.okta_auth as okta_module

    with (
        patch("streamlit.secrets", new={"auth": {"mode": "oidc"}}),
        patch.object(auth_module, "_handle_local_auth") as local_mock,
        patch.object(okta_module, "handle_oidc_auth") as oidc_mock,
    ):
        auth_module.check_authenticate()

    oidc_mock.assert_called_once()
    local_mock.assert_not_called()


def test_check_authenticate_routes_to_local_when_mode_is_local():
    """[auth].mode = 'local' is an explicit fallback → local path."""
    import utils.auth as auth_module
    import utils.okta_auth as okta_module

    with (
        patch("streamlit.secrets", new={"auth": {"mode": "local"}}),
        patch.object(auth_module, "_handle_local_auth") as local_mock,
        patch.object(okta_module, "handle_oidc_auth") as oidc_mock,
    ):
        auth_module.check_authenticate()

    local_mock.assert_called_once()
    oidc_mock.assert_not_called()
