import streamlit as st
from datetime import datetime, timedelta
from models.user import (
    verify_user_credentials, 
    get_user, 
    set_user_preferences_in_session_state
)

def check_authenticate():
    user_id = st.session_state.cookies.get("user_id")
    expiry_date_str = st.session_state.cookies.get("expiry_date")
    if user_id and expiry_date_str:
        expiry_date = datetime.fromisoformat(expiry_date_str)
        user = get_user(user_id)
        set_user_preferences_in_session_state(user)
        if datetime.now() < expiry_date:
            with st.sidebar:
                st.title(f"Welcome {user.first_name} {user.last_name}")
            logout = st.sidebar.button("Log Out", type="primary", use_container_width=True)
            if logout:
                st.session_state.cookies["user_id"] = ""
                st.session_state.cookies["expiry_date"] = ""
                # TODO: This doesnt work, how to delete cookies?
                # del st.session_state.cookies["user_id"]
                # del st.session_state.cookies["expiry_date"]
                st.session_state.cookies.save()
                st.rerun()
        else:
            show_login()
    else:
        show_login()

def show_login():
    st.title("🔓 Log In - Thrive AI")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Login", type="primary")

        if submit_button:
            if verify_user_credentials(username, password): # sets user_id in cookies
                expiry_date = datetime.now() + timedelta(days=1)
                st.session_state.cookies["expiry_date"] = expiry_date.isoformat()
                st.session_state.cookies.save()
                st.rerun()
            else:
                st.error("Incorrect username or password. Please try again.")
    st.stop()