import logging
from datetime import datetime, timedelta

import streamlit as st

from orm.functions import set_user_preferences_in_session_state, verify_user_credentials

logger = logging.getLogger(__name__)


def check_authenticate():
    try:
        user_id = st.session_state.cookies.get("user_id")
        expiry_date_str = st.session_state.cookies.get("expiry_date")
        if user_id and expiry_date_str:
            expiry_date = datetime.fromisoformat(expiry_date_str)
            user = set_user_preferences_in_session_state()

            # Ensure VannaService instance is cleared when switching users
            # This prevents the previous user's LLM selection from being displayed
            if "_vn_instance" in st.session_state and st.session_state._vn_instance is not None:
                # Check if the cached instance is for a different user
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
                        # Invalidate VannaService cache for this user
                        from utils.vanna_calls import VannaService
                        import json

                        user_id_for_cache = json.loads(st.session_state.cookies.get("user_id"))
                        user_role = st.session_state.get("user_role")
                        if user_id_for_cache and user_role is not None:
                            VannaService.invalidate_cache_for_user(str(user_id_for_cache), user_role)

                        # Clear cookies
                        st.session_state.cookies["user_id"] = ""
                        st.session_state.cookies["expiry_date"] = ""
                        # TODO: This doesnt work, how to delete cookies?
                        # del st.session_state.cookies["user_id"]
                        # del st.session_state.cookies["expiry_date"]
                        st.session_state.cookies.save()

                        # Clear session state (including LLM preferences and VannaService instance)
                        st.session_state.messages = []
                        st.session_state.selected_llm_provider = None
                        st.session_state.selected_llm_model = None
                        if "_vn_instance" in st.session_state:
                            st.session_state._vn_instance = None

                        st.rerun()
            else:
                show_login()
        else:
            show_login()
    except Exception as e:
        st.error(f"Error checking authentication: {e}")
        logger.error(f"Error checking authentication: {e}")


def show_login():
    try:
        # --- HIDE LOGIN NAVIGATION ---
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
        # --- HIDE LOGIN NAVIGATION ---
        st.title("🔓 Log In")

        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button("Login", type="primary")

            if submit_button:
                if verify_user_credentials(username, password):  # sets user_id in cookies
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
