import streamlit as st
from orm.functions import change_password, delete_all_messages
from utils.vanna_calls import (train_file, train_ddl, setup_vanna)

# Get the current user ID from session state cookies
user_id = st.session_state.cookies.get("user_id")
vn = setup_vanna()

def delete_all_training():
    training_data = vn.get_training_data()
    for index, row in training_data.iterrows():
        vn.remove_training_data(row["id"])
    st.toast("Training Data Deleted Successfully!")

st.title("User Settings")

with st.expander("Training Data", expanded=True):
    cols = st.columns((1, 1, 1))
    with cols[0]:
        st.button("Train DDL", on_click=lambda: train_ddl())
    with cols[1]:
        st.button("Train from FIle", on_click=lambda: train_file())
    with cols[2]:
        st.button("Remove All", on_click=lambda: delete_all_training())

    df =vn.get_training_data()

    colms = st.columns((1, 2, 2, 1, 1))
    fields = ["№", 'email', 'uid', 'verified', "action"]
    for col, field_name in zip(colms, fields):
        # header
        col.write(field_name)

    for index, row in df.iterrows():
        col1, col2, col3, col4 = st.columns((1, 2, 3, 1))
        col1.write(row['training_data_type'])
        col2.write(row['question'])
        col3.write(row['content'])
        button_phold = col4.empty() 
        do_action = button_phold.button(label="Delete", key=f"delete{row['id']}")
        if do_action:
            vn.remove_training_data(row['id'])
            st.toast("Training Data Deleted Successfully!")
            st.rerun()

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