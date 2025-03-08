import streamlit as st
from orm.functions import change_password, delete_all_messages
from utils.vanna_calls import (train, setup_vanna)

# Get the current user ID from session state cookies
user_id = st.session_state.cookies.get("user_id")
vn = setup_vanna()

def delete_message_by_id(message_id):
    vn.remove_training_data(message_id)
    st.toast("Training Data Deleted Successfully!")

st.title("User Settings")

with st.expander("Training Data", expanded=True):
    st.button("Train Vanna", on_click=lambda: train())
    # TODO: add a button to remove each row of training data
    # TODO: add a form for entering training data
    # TODO: breakout training to two commands, one for DDLs and one for fiile training
    st.dataframe(vn.get_training_data(), column_order=("question", "content", "training_data_type"), hide_index=True)

if user_id:
    with st.expander("Change Password"):
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

st.sidebar.button("Delete all message data", on_click=lambda: delete_all_messages(), use_container_width=True, type="primary")