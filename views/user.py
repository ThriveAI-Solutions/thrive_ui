import streamlit as st
from orm.functions import change_password, delete_all_messages
from utils.vanna_calls import (train)

# Streamlit application
st.title("Change Password")

# Get the current user ID from session state cookies
user_id = st.session_state.cookies.get("user_id")

if user_id:
    with st.form("change_password_form"):
        current_password = st.text_input("Current Password", type="password")
        new_password = st.text_input("New Password", type="password")
        confirm_new_password = st.text_input("Confirm New Password", type="password")
        submit_button = st.form_submit_button("Change Password")

        if submit_button:
            if new_password != confirm_new_password:
                st.error("New password and confirmation do not match.")
            else:
                if change_password(user_id, current_password, new_password):
                    st.success("Password changed successfully.")
                else:
                    st.error("Current password is incorrect.")
else:
    st.error("User not logged in.")

st.sidebar.button("Train Vanna", on_click=lambda: train(), use_container_width=True)
st.sidebar.button("Delete all message data", on_click=lambda: delete_all_messages(), use_container_width=True, type="primary")