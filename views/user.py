import logging
import io

import pandas as pd
import streamlit as st
from pandas import DataFrame

from orm.functions import change_password, create_user, delete_all_messages, get_all_user_roles
from orm.models import RoleTypeEnum, SessionLocal, User, UserRole
import hashlib
from sqlalchemy import func
from utils.chat_bot_helper import get_vn
from utils.vanna_calls import VannaService, train_ddl, train_file, training_plan
from utils.authentication_management import get_user_list_excel

# Get the current user ID from session state cookies
user_id = st.session_state.cookies.get("user_id")
# Get the current user role from session state (not cookies) and default to least privileged
user_role = st.session_state.get("user_role", RoleTypeEnum.PATIENT.value)
# Don't get training data at module load time - get it when rendering the page
# df = vn.get_training_data()

logging.debug(f"{st.session_state.to_dict()=}")

logger = logging.getLogger(__name__)


def import_users():
    """
    Import users from Excel file and save them to the database.
    Expected columns in Excel: username, password, first_name, last_name, role_name
    """
    try:
        import os
        
        # Use multiple path resolution strategies
        import pathlib
        
        # Try different path strategies
        possible_paths = [
            "./utils/config/user_list.xlsx",  # Relative to current directory
            "utils/config/user_list.xlsx",    # Without leading ./
            os.path.join(os.path.dirname(__file__), "..", "utils", "config", "user_list.xlsx"),  # Relative to this file
        ]
        
        file_path = None
        for path in possible_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                file_path = abs_path
                break
        
        current_dir = os.getcwd()
        
        st.write(f"🔍 **Debug Info:**")
        st.write(f"- Current working directory: `{current_dir}`")
        st.write(f"- Script file location: `{os.path.dirname(__file__)}`")
        
        if file_path:
            st.write(f"- Found file at: `{file_path}`")
        else:
            st.write("- **File not found in any expected location!**")
            st.write("- Tried paths:")
            for path in possible_paths:
                abs_path = os.path.abspath(path)
                exists = os.path.exists(abs_path)
                st.write(f"  - `{abs_path}` (exists: {exists})")
        
        if os.path.exists("./utils/config/"):
            files_in_dir = os.listdir("./utils/config/")
            st.write(f"- Files in utils/config/: `{files_in_dir}`")
        
        # Read the Excel file
        if file_path:
            df = get_user_list_excel(file_path)
        else:
            st.error("Excel file not found in any expected location!")
            return False
        
        if df is False or df is None:
            st.error(f"Failed to read user list Excel file at: `{file_path}`")
            return False
        
        if df.empty:
            st.warning("No users found in the Excel file.")
            return False
        
        # Track import results
        success_count = 0
        failed_count = 0
        failed_users = []
        
        with SessionLocal() as session:
            # Get all existing roles for mapping
            roles = session.query(UserRole).all()
            role_map = {role.role_name.lower(): role.id for role in roles}
            
            # Also check for common variations
            role_map.update({
                'administrator': role_map.get('admin', None),
                'physician': role_map.get('doctor', None),
            })
            
            # Process each row in the DataFrame
            for index, row in df.iterrows():
                try:
                    # Extract user data with default values if columns are missing
                    username = str(row.get('UserID', '')).strip()
                    password = str(row.get('start_password', '')).strip()
                    first_name = str(row.get('First Name ', '')).strip()  # Note the space after 'Name'
                    last_name = str(row.get('Last Name ', '')).strip()    # Note the space after 'Name'
                    role_name = 'Doctor'  # Default role since not in Excel
                    
                    # Skip rows with missing required data
                    if not username or not password:
                        failed_count += 1
                        failed_users.append(f"Row {index + 2}: Missing username or password")
                        continue
                    
                    # Check if user already exists
                    existing_user = session.query(User).filter(
                        func.lower(User.username) == username.lower()
                    ).first()
                    
                    if existing_user:
                        failed_count += 1
                        failed_users.append(f"{username}: Already exists")
                        continue
                    
                    # Map role name to role ID
                    role_id = role_map.get(role_name.lower())
                    if not role_id:
                        # Default to Patient role if role not found
                        default_role = session.query(UserRole).filter(
                            UserRole.role_name == "Patient"
                        ).first()
                        role_id = default_role.id if default_role else 1
                        logger.warning(f"Role '{role_name}' not found for user {username}, defaulting to Patient")
                    
                    # Hash the password using SHA-256 (matching existing pattern)
                    hashed_password = hashlib.sha256(password.encode()).hexdigest()
                    
                    # Create new user
                    new_user = User(
                        username=username,
                        password=hashed_password,
                        first_name=first_name if first_name else username,
                        last_name=last_name if last_name else '',
                        user_role_id=role_id,
                        show_sql=True,
                        show_table=True,
                        show_plotly_code=False,
                        show_chart=False,
                        show_question_history=True,
                        show_summary=True,
                        voice_input=False,
                        speak_summary=False,
                        show_suggested=False,
                        show_followup=False,
                        show_elapsed_time=True,
                        llm_fallback=False,
                        min_message_id=0
                    )
                    
                    session.add(new_user)
                    success_count += 1
                    
                except Exception as e:
                    failed_count += 1
                    failed_users.append(f"{row.get('username', f'Row {index + 2}')}: {str(e)}")
                    logger.error(f"Error importing user at row {index + 2}: {e}")
            
            # Commit all successful imports
            session.commit()
        
        # Display results
        if success_count > 0:
            st.success(f"✅ Successfully imported {success_count} user(s)")
        
        if failed_count > 0:
            st.warning(f"⚠️ Failed to import {failed_count} user(s)")
            if failed_users:
                with st.expander("Failed imports details"):
                    for failure in failed_users:
                        st.text(f"• {failure}")
        
        return success_count > 0
        
    except Exception as e:
        st.error(f"Error during user import: {e}")
        logger.error(f"Error during user import: {e}")
        return False


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


def export_training_data_to_csv():
    try:
        # Get training data with current user's role-based filtering
        vn = get_vn()
        training_data = vn.get_training_data()

        if isinstance(training_data, DataFrame) and not training_data.empty:
            # Create CSV buffer
            csv_buffer = io.StringIO()
            training_data.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()

            # Generate filename with timestamp
            import datetime

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_data_{timestamp}.csv"

            # Provide download button
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=filename,
                mime="text/csv",
                help="Download training data as CSV file",
            )
            st.toast("CSV export ready for download!")
        else:
            st.warning("No training data available to export.")
    except Exception as e:
        st.error(f"An error occurred during CSV export: {e}")
        logger.error(f"CSV export error: {e}")


def import_training_data_from_csv(uploaded_file):
    try:
        # Read the uploaded CSV file
        df = pd.read_csv(uploaded_file)

        # Validate required columns
        required_columns = ["training_data_type", "question", "content"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
            return

        # Get Vanna instance
        vn = get_vn()
        success_count = 0
        error_count = 0

        # Process each row
        progress_bar = st.progress(0)
        total_rows = len(df)

        for index, row in df.iterrows():
            try:
                # Update progress
                progress_bar.progress((index + 1) / total_rows)

                # Skip rows with missing essential data
                if pd.isna(row["content"]) or str(row["content"]).strip() == "":
                    error_count += 1
                    continue

                training_type = str(row["training_data_type"]).lower()
                question = str(row["question"]) if not pd.isna(row["question"]) else None
                content = str(row["content"])

                # Train based on the data type
                if training_type == "sql" and question and question.strip():
                    # SQL training data
                    vn.train(question=question, sql=content)
                elif training_type in ["ddl", "documentation"]:
                    # Documentation or DDL training data
                    vn.train(documentation=content)
                else:
                    # Default to documentation if unclear
                    vn.train(documentation=content)

                success_count += 1

            except Exception as row_error:
                error_count += 1
                logger.error(f"Error processing row {index}: {row_error}")

        progress_bar.empty()

        # Show results
        if success_count > 0:
            st.success(f"Successfully imported {success_count} training entries!")
            if error_count > 0:
                st.warning(f"{error_count} entries failed to import. Check logs for details.")
            st.rerun()
        else:
            st.error("No training data was successfully imported.")

    except Exception as e:
        st.error(f"An error occurred during CSV import: {e}")
        logger.error(f"CSV import error: {e}")


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

tabs = ["Change Password", ""]
if st.session_state.cookies.get("role_name") == "Admin":
    tabs = ["Change Password", "Training Data"]
tab1, tab2 = st.tabs(tabs)

with tab1:
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

with tab2:
    if st.session_state.cookies.get("role_name") == "Admin":
        cols = st.columns((0.15, 0.25, 0.15, 0.15, 0.15, 0.25, 0.15, 0.15))
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
        with cols[7]:
            if st.button("Export CSV"):
                export_training_data_to_csv()
        
        # Add user import section
        st.divider()
        st.subheader("Bulk User Import")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info("📁 Import users from Excel file: `./utils/config/user_list.xlsx`")
            st.caption("Expected columns: UserID, start_password, 'First Name ', 'Last Name '")
        with col2:
            if st.button("Import Users", type="primary", help="Import users from ./utils/config/user_list.xlsx"):
                import_users()

        # Add CSV import functionality
        st.divider()
        st.subheader("Import Training Data from CSV")
        uploaded_file = st.file_uploader(
            "Choose a CSV file to upload training data",
            type=["csv"],
            help="Upload a CSV file with columns: training_data_type, question, content",
        )

        if uploaded_file is not None:
            # Show file details
            st.info(f"📁 File: {uploaded_file.name} ({uploaded_file.size} bytes)")

            # Add confirmation button
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📤 Import Training Data", type="primary"):
                    with st.spinner("Importing training data..."):
                        import_training_data_from_csv(uploaded_file)
            with col2:
                if st.button("❌ Cancel"):
                    st.rerun()

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

if st.session_state.cookies.get("role_name") == "Admin":
    st.sidebar.button("Delete all message data", on_click=delete_all_messages, use_container_width=True, type="primary")
