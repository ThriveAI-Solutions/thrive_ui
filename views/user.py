import hashlib
import io
import logging

import pandas as pd
import streamlit as st
from pandas import DataFrame
from sqlalchemy import func

from orm.functions import (
    admin_change_password,
    change_password,
    create_user,
    delete_all_messages,
    delete_user,
    get_all_user_roles,
    get_all_users,
    get_user_daily_stats,
    get_user_questions_page,
    get_user_recent_questions,
    get_user_stats_for_all_users,
    update_user,
    update_user_preferences,
)
from orm.models import RoleTypeEnum, SessionLocal, User, UserRole
from utils.authentication_management import get_user_list_excel
from utils.chat_bot_helper import get_vn
from utils.enums import ThemeType
from utils.vanna_calls import train_ai_documentation, train_ddl, train_enhanced_schema, train_file, training_plan

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

        # Use robust path resolution
        possible_paths = [
            "./utils/config/user_list.xlsx",
            "utils/config/user_list.xlsx",
            os.path.join(os.path.dirname(__file__), "..", "utils", "config", "user_list.xlsx"),
        ]

        file_path = None
        for path in possible_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                file_path = abs_path
                break

        if not file_path:
            st.error("Excel file not found. Please ensure 'utils/config/user_list.xlsx' exists.")
            return False

        # Read the Excel file (with CSV fallback)
        try:
            df = get_user_list_excel(file_path)
        except:
            # Try CSV fallback
            csv_path = file_path.replace(".xlsx", ".csv")
            if os.path.exists(csv_path):
                st.info("Using CSV fallback file")
                df = pd.read_csv(csv_path)
            else:
                st.error("Excel file requires openpyxl library. Please install openpyxl or provide a CSV file.")
                return False

        if df is False or df is None:
            st.error(f"Failed to read user list Excel file. Please check the file format and dependencies.")
            return False

        if df.empty:
            st.warning("Excel file contains no data")
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
            role_map.update(
                {
                    "administrator": role_map.get("admin", None),
                    "physician": role_map.get("doctor", None),
                }
            )

            # Process each row in the DataFrame
            for index, row in df.iterrows():
                try:
                    # Extract user data with default values if columns are missing
                    username = str(row.get("UserID", "")).strip()
                    password = str(row.get("start_password", "")).strip()
                    first_name = str(row.get("First Name ", "")).strip()  # Note the space after 'Name'
                    last_name = str(row.get("Last Name ", "")).strip()  # Note the space after 'Name'
                    role_name = "Doctor"  # Default role since not in Excel

                    # Skip rows with missing required data
                    if not username or not password:
                        failed_count += 1
                        failed_users.append(f"Row {index + 2}: Missing username or password")
                        continue

                    # Check if user already exists
                    existing_user = session.query(User).filter(func.lower(User.username) == username.lower()).first()

                    if existing_user:
                        failed_count += 1
                        failed_users.append(f"{username}: Already exists")
                        continue

                    # Map role name to role ID
                    role_id = role_map.get(role_name.lower())
                    if not role_id:
                        # Default to Patient role if role not found
                        default_role = session.query(UserRole).filter(UserRole.role_name == "Patient").first()
                        role_id = default_role.id if default_role else 1
                        logger.warning(f"Role '{role_name}' not found for user {username}, defaulting to Patient")

                    # Hash the password using SHA-256 (matching existing pattern)
                    hashed_password = hashlib.sha256(password.encode()).hexdigest()

                    # Create new user
                    new_user = User(
                        username=username,
                        password=hashed_password,
                        first_name=first_name if first_name else username,
                        last_name=last_name if last_name else "",
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
                        min_message_id=0,
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

tabs = ["Change Password"]
if st.session_state.get("user_role") == RoleTypeEnum.ADMIN.value:
    tabs = ["Change Password", "Training Data", "Manage Users"]
tab_objects = st.tabs(tabs)
tab1 = tab_objects[0]
tab2 = tab_objects[1] if len(tab_objects) > 1 else None
tab3 = tab_objects[2] if len(tab_objects) > 2 else None

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
    if tab2 and st.session_state.get("user_role") == RoleTypeEnum.ADMIN.value:
        # Enhanced Training Section (prominent placement)
        st.subheader("🚀 Automatic Schema Enrichment")
        st.caption("Automatically extract and train column statistics, relationships, and semantic information from your database.")

        with st.expander("Enhanced Training Options", expanded=False):
            col_stats, col_rels, col_views = st.columns(3)
            with col_stats:
                include_stats = st.checkbox("Include Column Statistics", value=True, help="Extract min/max/distinct counts, null ratios, and top values")
            with col_rels:
                include_rels = st.checkbox("Include Relationships", value=True, help="Discover FK relationships and infer implicit relationships by naming patterns")
            with col_views:
                include_views = st.checkbox("Include View Definitions", value=True, help="Extract and document view SQL definitions")

            sample_limit = st.number_input("Sample Limit (rows per column)", min_value=100, max_value=10000, value=1000, help="Maximum rows to sample per column for statistics")

            if st.button("🔬 Run Enhanced Training", type="primary", help="Analyze database schema and automatically train RAG with enriched metadata"):
                train_enhanced_schema(
                    include_statistics=include_stats,
                    include_relationships=include_rels,
                    include_view_definitions=include_views,
                    sample_limit=sample_limit,
                )

        st.divider()

        # Standard Training Section
        st.subheader("Standard Training")
        cols = st.columns((0.15, 0.20, 0.15, 0.15, 0.15, 0.25, 0.15, 0.15))
        with cols[0]:
            st.button("Train DDL", on_click=train_ddl)
        with cols[1]:
            st.button("AI Generate Docs", on_click=train_ai_documentation)
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

        # (Moved Bulk User Import to Manage Users tab)

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

        if isinstance(df, DataFrame) and not df.empty:
            st.divider()
            st.subheader("Training Data")

            # Prepare display dataframe with renamed columns for clarity
            display_df = df[["id", "training_data_type", "question", "content"]].copy()
            display_df.columns = ["ID", "Type", "Question", "Content"]

            # Use data_editor for interactive grid with selection for deletion
            if st.session_state.cookies.get("role_name") == "Admin":
                # Add selection column for bulk delete
                display_df.insert(0, "Select", False)

                edited_df = st.data_editor(
                    display_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Select": st.column_config.CheckboxColumn(
                            "Delete?",
                            help="Select rows to delete",
                            default=False,
                            width="small",
                        ),
                        "ID": st.column_config.TextColumn(
                            "ID",
                            width="small",
                            disabled=True,
                        ),
                        "Type": st.column_config.TextColumn(
                            "Type",
                            width="small",
                            disabled=True,
                        ),
                        "Question": st.column_config.TextColumn(
                            "Question",
                            width="medium",
                            disabled=True,
                        ),
                        "Content": st.column_config.TextColumn(
                            "Content",
                            width="large",
                            disabled=True,
                        ),
                    },
                    num_rows="fixed",
                    key="training_data_grid",
                )

                # Delete selected rows
                selected_rows = edited_df[edited_df["Select"]]
                if not selected_rows.empty:
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        if st.button(
                            f"🗑️ Delete {len(selected_rows)} Selected",
                            type="primary",
                            help="Delete all selected training data entries",
                        ):
                            vn = get_vn()
                            for _, row in selected_rows.iterrows():
                                vn.remove_from_training(row["ID"])
                            st.toast(f"Deleted {len(selected_rows)} training data entries!")
                            st.rerun()
                    with col2:
                        st.caption(f"Selected {len(selected_rows)} of {len(display_df)} entries for deletion")
            else:
                # Non-admin view: read-only dataframe
                st.dataframe(
                    display_df[["ID", "Type", "Question", "Content"]],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "ID": st.column_config.TextColumn("ID", width="small"),
                        "Type": st.column_config.TextColumn("Type", width="small"),
                        "Question": st.column_config.TextColumn("Question", width="medium"),
                        "Content": st.column_config.TextColumn("Content", width="large"),
                    },
                )

            # Show record count
            st.caption(f"Total: {len(df)} training data entries")
        else:
            st.info("No training data available.")

if tab3 and st.session_state.get("user_role") == RoleTypeEnum.ADMIN.value:
    with tab3:
        st.subheader("Manage Users")
        st.caption("Create, edit, or remove users. View settings and activity stats.")

        roles = get_all_user_roles()
        role_id_by_name = {name: rid for rid, name, _ in roles}
        role_names = [name for rid, name, _ in roles]

        stats_map = get_user_stats_for_all_users()

        left, right = st.columns([1, 2])
        with left:
            st.markdown("**Create New User**")
            with st.form("create_user_form", clear_on_submit=True):
                cu_username = st.text_input("Username")
                cu_password = st.text_input("Temporary Password", type="password")
                cu_first = st.text_input("First Name")
                cu_last = st.text_input("Last Name")
                cu_role_name = st.selectbox(
                    "Role", options=role_names, index=role_names.index("Patient") if "Patient" in role_names else 0
                )
                theme_options = [t.value for t in ThemeType]
                cu_theme = st.selectbox("Theme", options=theme_options, index=0)
                submitted = st.form_submit_button("Create User", type="primary")
                if submitted:
                    if not cu_username or not cu_password or not cu_first:
                        st.error("Please provide username, password, and first name.")
                    else:
                        ok = create_user(
                            cu_username, cu_password, cu_first, cu_last, role_id_by_name.get(cu_role_name), cu_theme
                        )
                        if ok:
                            st.success("User created.")
                            st.rerun()
                        else:
                            st.error("Failed to create user. Username may already exist.")

            st.divider()
            st.markdown("**Bulk User Import**")
            st.info("📁 Import users from Excel file: `./utils/config/user_list.xlsx`")
            st.caption("Expected columns: UserID, start_password, 'First Name ', 'Last Name '")
            if st.button("Import Users", type="primary", help="Import users from ./utils/config/user_list.xlsx"):
                import_users()

        with right:
            st.markdown("**Edit Existing User**")
            users = get_all_users()
            search = st.text_input("Search users", placeholder="Filter by username or name…")
            if search:
                s = search.lower()
                users = [
                    u
                    for u in users
                    if s in u["username"].lower() or s in f"{u['first_name']} {u['last_name']}`".lower()
                ]

            if users:
                user_options = {
                    f"{u['username']} ({u['first_name']} {u['last_name']}) - {u['role_name']}": u for u in users
                }
                selected_label = st.selectbox("Select a user", options=list(user_options.keys()))
                selected = user_options[selected_label]

                # Load full user for preferences
                with SessionLocal() as session:
                    db_user = session.query(User).filter(User.id == selected["id"]).first()

                st.divider()
                # Time series chart spanning the full right pane width
                range_choice = st.radio("Range", options=["7 days", "30 days"], horizontal=True, key="stats_range")
                days = 7 if range_choice.startswith("7") else 30
                daily = get_user_daily_stats(selected["id"], days=days)
                if daily:
                    import plotly.express as px

                    chart_df = pd.DataFrame(daily)
                    chart_df["date"] = pd.to_datetime(chart_df["date"])
                    melted = chart_df.melt(
                        id_vars=["date"],
                        value_vars=["questions", "charts", "errors", "dataframes", "summaries"],
                        var_name="metric",
                        value_name="count",
                    )
                    fig = px.line(melted, x="date", y="count", color="metric", markers=True)
                    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), legend_title_text="")
                    st.plotly_chart(fig, use_container_width=True)

                # Questions table and download (full width under chart)
                with st.expander("Recent Questions"):
                    # Initialize and apply pending navigation before rendering widgets
                    # Reset pagination when switching selected user
                    if st.session_state.get("q_selected_user_id") != selected["id"]:
                        st.session_state["q_selected_user_id"] = selected["id"]
                        st.session_state["q_page_num"] = 1
                    if "q_page_num" not in st.session_state:
                        st.session_state["q_page_num"] = 1
                    if "q_page_bump" in st.session_state:
                        st.session_state["q_page_num"] = max(
                            1, int(st.session_state.get("q_page_num", 1)) + int(st.session_state["q_page_bump"])
                        )
                        del st.session_state["q_page_bump"]

                    colq1, colq2, colq3 = st.columns([1, 1, 6])
                    with colq1:
                        page_size = st.selectbox("Page size", options=[25, 50, 100], index=1, key="q_page_size")
                    with colq2:
                        st.number_input("Page", min_value=1, step=1, key="q_page_num")
                        page = int(st.session_state["q_page_num"])

                    page_data = get_user_questions_page(selected["id"], page=int(page), page_size=int(page_size))
                    items = page_data.get("items", [])
                    total = page_data.get("total", 0)

                    if items:
                        qdf = pd.DataFrame(items)
                        qdf.rename(
                            columns={
                                "question": "Question",
                                "created_at": "Asked At",
                                "status": "Status",
                                "elapsed_seconds": "Elapsed (s)",
                            },
                            inplace=True,
                        )
                        st.dataframe(qdf, use_container_width=True, hide_index=True)

                        # Pagination controls
                        total_pages = max(1, (total + int(page_size) - 1) // int(page_size))
                        cprev, cinfo, cnext = st.columns([1, 3, 1])
                        with cprev:
                            st.button(
                                "Prev",
                                disabled=int(page) <= 1,
                                on_click=lambda: st.session_state.update({"q_page_bump": -1}),
                            )
                        with cinfo:
                            st.caption(f"Page {int(page)} of {total_pages} • {total} total")
                        with cnext:
                            st.button(
                                "Next",
                                disabled=int(page) >= total_pages,
                                on_click=lambda: st.session_state.update({"q_page_bump": 1}),
                            )

                        # Download all recent questions as CSV including status/elapsed when available
                        all_rows = []
                        # Fetch in chunks if needed
                        all_page_size = 1000
                        remaining = total
                        page_iter = 1
                        while remaining > 0 and page_iter <= 100:  # safety cap
                            p = get_user_questions_page(selected["id"], page=page_iter, page_size=all_page_size)
                            all_rows.extend(p.get("items", []))
                            if len(p.get("items", [])) < all_page_size:
                                break
                            remaining -= len(p.get("items", []))
                            page_iter += 1
                        if all_rows:
                            all_df = pd.DataFrame(all_rows)
                            all_df.rename(
                                columns={
                                    "question": "Question",
                                    "created_at": "Asked At",
                                    "status": "Status",
                                    "elapsed_seconds": "Elapsed (s)",
                                },
                                inplace=True,
                            )
                            csv_bytes = all_df.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                label="Download all questions (.csv)",
                                data=csv_bytes,
                                file_name=f"{selected['username']}_questions.csv",
                                mime="text/csv",
                            )
                    else:
                        st.info("No questions found.")

                c1, c2 = st.columns([2, 1])
                with c1:
                    st.markdown("**Profile**")
                    nu_username = st.text_input("Username", value=selected["username"])
                    nu_first = st.text_input("First Name", value=selected["first_name"])
                    nu_last = st.text_input("Last Name", value=selected["last_name"])
                    nu_role_name = st.selectbox(
                        "Role",
                        options=role_names,
                        index=role_names.index(selected["role_name"]) if selected["role_name"] in role_names else 0,
                    )
                    theme_options = [t.value for t in ThemeType]
                    current_theme = selected.get("theme", ThemeType.HEALTHELINK.value)
                    nu_theme = st.selectbox(
                        "Theme",
                        options=theme_options,
                        index=theme_options.index(current_theme) if current_theme in theme_options else 0,
                    )
                    cols = st.columns(2)
                    with cols[0]:
                        if st.button("Save Profile", key="save_profile", type="primary"):
                            ok = update_user(
                                selected["id"],
                                nu_username,
                                nu_first,
                                nu_last,
                                role_id_by_name.get(nu_role_name),
                                nu_theme,
                            )
                            if ok:
                                st.success("Profile updated.")
                                st.rerun()
                            else:
                                st.error("Failed to update profile.")
                    with cols[1]:
                        new_pw = st.text_input("Set New Password", type="password", key="admin_pw")
                        if st.button("Update Password", key="update_pw"):
                            if not new_pw:
                                st.error("Enter a new password.")
                            else:
                                if admin_change_password(selected["id"], new_pw):
                                    st.success("Password updated.")
                                else:
                                    st.error("Failed to update password.")

                with c2:
                    st.markdown("**Stats**")
                    s = stats_map.get(
                        selected["id"], {"questions": 0, "charts": 0, "errors": 0, "dataframes": 0, "summaries": 0}
                    )
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Questions", s["questions"])
                    m2.metric("Charts", s["charts"])
                    m3.metric("Errors", s["errors"])
                    m4, m5 = st.columns(2)
                    m4.metric("DataFrames", s["dataframes"])
                    m5.metric("Summaries", s["summaries"])

                st.divider()
                st.markdown("**Preferences**")
                if db_user:
                    pref_cols = st.columns(3)
                    with pref_cols[0]:
                        p_show_sql = st.checkbox("Show SQL", value=db_user.show_sql)
                        p_show_table = st.checkbox("Show Table", value=db_user.show_table)
                        p_plotly = st.checkbox("Show Plotly Code", value=db_user.show_plotly_code)
                        p_chart = st.checkbox("Show Chart", value=db_user.show_chart)
                    with pref_cols[1]:
                        p_history = st.checkbox("Show Question History", value=db_user.show_question_history)
                        p_summary = st.checkbox("Show Summary", value=db_user.show_summary)
                        p_voice = st.checkbox("Voice Input", value=db_user.voice_input)
                        p_speak = st.checkbox("Speak Summary", value=db_user.speak_summary)
                    with pref_cols[2]:
                        p_suggested = st.checkbox("Show Suggested", value=db_user.show_suggested)
                        p_followup = st.checkbox("Show Follow-up", value=db_user.show_followup)
                        p_elapsed = st.checkbox("Show Elapsed Time", value=db_user.show_elapsed_time)
                        p_llm = st.checkbox("LLM Fallback", value=db_user.llm_fallback)

                    if st.button("Save Preferences", key="save_prefs", type="primary"):
                        ok = update_user_preferences(
                            selected["id"],
                            show_sql=p_show_sql,
                            show_table=p_show_table,
                            show_plotly_code=p_plotly,
                            show_chart=p_chart,
                            show_question_history=p_history,
                            show_summary=p_summary,
                            voice_input=p_voice,
                            speak_summary=p_speak,
                            show_suggested=p_suggested,
                            show_followup=p_followup,
                            show_elapsed_time=p_elapsed,
                            llm_fallback=p_llm,
                        )
                        if ok:
                            st.success("Preferences saved.")
                        else:
                            st.error("Failed to save preferences.")

                st.divider()
                st.markdown("**Danger Zone**")
                colz = st.columns([1, 2])
                with colz[0]:
                    confirm = st.text_input("Type username to confirm delete", key="confirm_del")
                with colz[1]:
                    if st.button("Delete User", type="primary"):
                        if confirm != selected["username"]:
                            st.error("Confirmation does not match username.")
                        else:
                            if delete_user(selected["id"]):
                                st.success("User deleted.")
                                st.rerun()
                            else:
                                st.error("Failed to delete user.")
            else:
                st.info("No users found.")

    st.sidebar.button("Delete all message data", on_click=delete_all_messages, use_container_width=True, type="primary")
