from datetime import datetime, timedelta

import streamlit as st

import utils.okta_auth as okta_auth
from orm.functions import set_user_preferences_in_session_state, verify_user_credentials
from utils.quick_logger import get_logger

logger = get_logger(__name__)


def check_authenticate():
    """Dispatcher: route to OIDC handler or existing local-cookie path.

    The mode is determined by `[auth].mode` in secrets.toml. Absent or
    `mode = "local"` → existing username/password flow. `mode = "oidc"`
    → Streamlit-native OIDC via st.login() / st.user.

    Both paths leave session state in the same shape (cookies['user_id'],
    cookies['role_name'], session_state.user_role, session_state.username,
    plus all preference flags).

    Note: imports utils.okta_auth as a module (not the names) so test
    mocks of utils.okta_auth.handle_oidc_auth are visible at call time.
    """
    if okta_auth.is_oidc_mode():
        okta_auth.handle_oidc_auth()
    else:
        _handle_local_auth()


def _handle_local_auth():
    """Existing local-cookie auth path. Behavior unchanged from pre-OIDC."""
    try:
        user_id = st.session_state.cookies.get("user_id")
        expiry_date_str = st.session_state.cookies.get("expiry_date")
        if user_id and expiry_date_str:
            expiry_date = datetime.fromisoformat(expiry_date_str)
            user = set_user_preferences_in_session_state()

            # Cookie present but its user_id no longer resolves to a user
            # (stale cookie after a DB recreate/migrate). Clear the stale
            # cookie and fall back to the login form rather than dereferencing
            # a None user below.
            if user is None:
                st.session_state.cookies["user_id"] = ""
                st.session_state.cookies["expiry_date"] = ""
                st.session_state.cookies.save()
                _show_local_login()
                return

            # Ensure VannaService instance is cleared when switching users.
            if "_vn_instance" in st.session_state and st.session_state._vn_instance is not None:
                import json

                cached_user_id = getattr(st.session_state._vn_instance, "user_id", None)
                current_user_id = str(json.loads(user_id))
                if cached_user_id != current_user_id:
                    st.session_state._vn_instance = None

            if datetime.now() < expiry_date:
                cols = st.sidebar.columns([0.7, 0.3], vertical_alignment="bottom")
                with cols[0]:
                    st.title(f"Welcome {user.first_name} {user.last_name}")
                with cols[1]:
                    logout = st.button("Log Out")
                    if logout:
                        # Invalidate VannaService cache for this user.
                        from utils.vanna_calls import VannaService
                        import json

                        user_id_for_cache = json.loads(st.session_state.cookies.get("user_id"))
                        user_role = st.session_state.get("user_role")
                        if user_id_for_cache and user_role is not None:
                            VannaService.invalidate_cache_for_user(str(user_id_for_cache), user_role)

                        # Clear cookies.
                        st.session_state.cookies["user_id"] = ""
                        st.session_state.cookies["expiry_date"] = ""
                        st.session_state.cookies.save()

                        # Clear session state.
                        st.session_state.messages = []
                        st.session_state.selected_llm_provider = None
                        st.session_state.selected_llm_model = None
                        if "_vn_instance" in st.session_state:
                            st.session_state._vn_instance = None

                        st.rerun()
            else:
                _show_local_login()
        else:
            _show_local_login()
    except Exception as e:
        st.error(f"Error checking authentication: {e}")
        logger.error(f"Error checking authentication: {e}")


def _show_local_login():
    """Render the local username/password login form."""
    try:
        st.markdown(
            """
        <style>
            [data-testid="stSidebarCollapsedControl"], [data-testid="stSidebar"] {
                display: none
            }
        </style>
        """,
            unsafe_allow_html=True,
        )
        st.title("🔓 Log In")

        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button("Login", type="primary")

            if submit_button:
                if verify_user_credentials(username, password):
                    expiry_date = datetime.now() + timedelta(hours=8)
                    st.session_state.cookies["expiry_date"] = expiry_date.isoformat()
                    st.session_state.cookies.save()
                    st.rerun()
                else:
                    st.error("Incorrect username or password. Please try again.")
        st.stop()
    except Exception as e:
        st.error(f"Error showing login: {e}")
        logger.error(f"Error showing login: {e}")


# Backwards-compat alias: external imports of show_login still work.
show_login = _show_local_login
