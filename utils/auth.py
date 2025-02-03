import streamlit as st
import json
from datetime import datetime, timedelta
# from sqlalchemy.orm import Session
from models.database import User, SessionLocal
import hashlib

def verify_user_credentials(username: str, password: str) -> bool:
    # Create a new database session
    session = SessionLocal()

    # Hash the password using SHA-256
    hashed_password = hashlib.sha256(password.encode()).hexdigest()

    # Query to check if the username and hashed password exist in the users table
    user = session.query(User).filter(User.username == username, User.password == hashed_password).first()
    st.session_state.cookies["user"] = json.dumps(user.to_dict())

    # Close the database session
    session.close()

    # Return True if the user exists, otherwise return False
    return user is not None

def check_authenticate():
    user = st.session_state.cookies.get("user")
    expiry_date_str = st.session_state.cookies.get("expiry_date")
    if user and expiry_date_str:
        expiry_date = datetime.fromisoformat(expiry_date_str)
        if datetime.now() < expiry_date:
            user = json.loads(user)
            with st.sidebar:
                st.title(f"Welcome {user['first_name']} {user['last_name']}")
            logout = st.sidebar.button("Log Out", type="primary", use_container_width=True)
            if logout:
                st.session_state.cookies["user"] = ""
                st.session_state.cookies["expiry_date"] = ""
                # This doesnt work, how to delete cookies?
                # del st.session_state.cookies["user"]
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
            if verify_user_credentials(username, password):
                expiry_date = datetime.now() + timedelta(days=1)
                st.session_state.cookies["expiry_date"] = expiry_date.isoformat()
                st.session_state.cookies.save()
                st.rerun()
            else:
                st.error("Incorrect username or password. Please try again.")
    st.stop()