import logging

import streamlit as st
from pandas import DataFrame

from orm.functions import change_password, delete_all_messages
from orm.models import RoleTypeEnum
from utils.chat_bot_helper import get_vn
from utils.vanna_calls import VannaService, train_ddl, train_file, training_plan

# Get the current user ID from session state cookies
user_id = st.session_state.cookies.get("user_id")
# Get the current user role from session state (not cookies) and default to least privileged
user_role = st.session_state.get("user_role", RoleTypeEnum.PATIENT.value)
# Don't get training data at module load time - get it when rendering the page
# df = vn.get_training_data()

logging.debug(f"{st.session_state.to_dict()=}")

logger = logging.getLogger(__name__)


def delete_all_training():
    try:
        # Get training data with role-based filtering - users can only delete what they can see
        vn = get_vn()
        training_data = vn.get_training_data()
        for index, row in training_data.iterrows():
            vn.remove_from_training(row["id"])
        st.toast("Training Data Deleted Successfully!")
    except Exception as e:
        st.error(f"An error occurred: {e}")
        logger.error(f"An error occurred: {e}")


@st.dialog("Cast your vote")
def pop_train(type):
    try:
        if type == "sql":
            with st.form("add_training_data"):
                question = st.text_input("Question")
                content1 = st.text_input("Sql")
                if st.form_submit_button("Add"):
                    if question == "" or content1 == "":
                        st.error("Please fill all fields.")
                    else:
                        get_vn().train(question=question, sql=content1)
                        st.toast("Training Data Added Successfully!")
                        st.rerun()
        else:
            with st.form("add_training_data"):
                content2 = st.text_input("Documentation")
                if st.form_submit_button("Add"):
                    if content2 == "":
                        st.error("Please fill all fields.")
                    else:
                        get_vn().train(documentation=content2)
                        st.toast("Training Data Added Successfully!")
                        st.rerun()
    except Exception as e:
        st.error(f"An error occurred: {e}")
        logger.error(f"An error occurred: {e}")


st.title("User Settings")

tab1, tab2 = st.tabs(["Training Data", "Change Password"])

with tab1:
    if st.session_state.cookies.get("role_name") == "Admin":
        cols = st.columns((0.2, 0.3, 0.2, 0.2, 0.2, 0.3, 0.2))
        with cols[0]:
            st.button("Train DDL", on_click=train_ddl)
        with cols[1]:
            st.button("DDL Describe", on_click=lambda: train_ddl(describe_ddl=True))
        with cols[2]:
            st.button("Train Plan", on_click=training_plan)
        with cols[3]:
            st.button("Train FIle", on_click=train_file)
        with cols[4]:
            if st.button("Add Sql"):
                pop_train("sql")
        with cols[5]:
            if st.button("Add Documentation"):
                pop_train("documentation")
        with cols[6]:
            st.button("Remove All", type="primary", on_click=delete_all_training)

    # Get training data with current user's role-based filtering
    df = get_vn().get_training_data()

    colms = st.columns((1, 2, 3, 1))
    fields = ["Type", "Question", "Sql", "Action"]
    for col, field_name in zip(colms, fields):
        # header
        col.write(field_name)

    if isinstance(df, DataFrame) and not df.empty:
        for index, row in df.iterrows():
            col1, col2, col3, col4 = st.columns((1, 2, 3, 1))
            col1.write(row["training_data_type"])
            col2.write(row["question"])
            col3.write(row["content"])
            if st.session_state.cookies.get("role_name") == "Admin":
                button_phold = col4.empty()
                do_action = button_phold.button(label="Delete", type="primary", key=f"delete{row['id']}")
                if do_action:
                    get_vn().remove_from_training(row["id"])
                    st.toast("Training Data Deleted Successfully!")
                    st.rerun()
with tab2:
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

if st.session_state.cookies.get("role_name") == "Admin":
    st.sidebar.button("Delete all message data", on_click=delete_all_messages, use_container_width=True, type="primary")
