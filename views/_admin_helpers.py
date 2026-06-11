"""Shared helpers for the Admin sub-tab modules (Epic #144).

Centralizes the @st.dialog helpers, CSV import/export utilities, and other
small functions previously defined inline in views/user.py. Consumed by
views/admin_users.py and views/admin_training.py so neither has to duplicate
the dialog definitions.
"""

import io

import pandas as pd
import streamlit as st
from pandas import DataFrame

from orm.functions import (
    UserValidationError,
    admin_change_password,
    create_user,
    delete_user,
    get_all_user_roles,
    get_users_for_export,
)
from orm.models import SessionLocal, UserRole
from utils.authentication_management import get_user_list_excel
from utils.chat_bot_helper import get_vn
from utils.enums import user_selectable_themes
from utils.quick_logger import get_logger

logger = get_logger(__name__)


def _cell_text(raw) -> str:
    """Coerce a pandas cell to a stripped string. Returns '' for NaN/None."""
    if raw is None:
        return ""
    if isinstance(raw, float) and pd.isna(raw):
        return ""
    if pd.isna(raw):
        return ""
    return str(raw).strip()


def import_users():
    """Import users from Excel file and save them to the database.

    Expected columns: UserID, start_password, First Name, Last Name, Email, Organization.
    Validation + persistence is delegated to ``orm.functions.create_user``.
    """
    try:
        import os

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

        try:
            df = get_user_list_excel(file_path)
        except Exception:
            csv_path = file_path.replace(".xlsx", ".csv")
            if os.path.exists(csv_path):
                st.info("Using CSV fallback file")
                df = pd.read_csv(csv_path)
            else:
                st.error("Excel file requires openpyxl library. Please install openpyxl or provide a CSV file.")
                return False

        if df is False or df is None:
            st.error("Failed to read user list Excel file. Please check the file format and dependencies.")
            return False

        if df.empty:
            st.warning("Excel file contains no data")
            return False

        success_count = 0
        failed_count = 0
        failed_users = []

        with SessionLocal() as session:
            roles = session.query(UserRole).all()
            role_map = {role.role_name.lower(): role.id for role in roles}
            role_map.update(
                {
                    "administrator": role_map.get("admin", None),
                    "physician": role_map.get("doctor", None),
                }
            )
            _patient_fallback_role_id = role_map.get("patient", 1)

        for index, row in df.iterrows():
            try:
                username = _cell_text(row.get("UserID"))
                password = _cell_text(row.get("start_password"))
                first_name = _cell_text(row.get("First Name "))
                last_name = _cell_text(row.get("Last Name "))
                email = _cell_text(row.get("Email", row.get("email")))
                organization = _cell_text(row.get("Organization", row.get("organization")))
                role_name = "Doctor"

                if not username or not password:
                    failed_count += 1
                    failed_users.append(f"Row {index + 2}: Missing username or password")
                    continue
                if not email:
                    failed_count += 1
                    failed_users.append(f"Row {index + 2} ({username}): Missing email")
                    continue
                if not organization:
                    failed_count += 1
                    failed_users.append(f"Row {index + 2} ({username}): Missing organization")
                    continue

                role_id = role_map.get(role_name.lower())
                if not role_id:
                    role_id = _patient_fallback_role_id
                    logger.warning(f"Role '{role_name}' not found for user {username}, defaulting to Patient")

                try:
                    ok = create_user(
                        username,
                        password,
                        first_name if first_name else username,
                        last_name if last_name else "",
                        role_id,
                        email=email,
                        organization=organization,
                    )
                except UserValidationError as ve:
                    # Required-field validation per Epic #179. Surface the
                    # specific missing fields rather than the generic "rejected"
                    # message so the admin can fix the CSV row directly.
                    failed_count += 1
                    failed_users.append(
                        f"{username}: missing or invalid required field(s) — {', '.join(ve.missing_fields)}"
                    )
                    continue
                if ok:
                    success_count += 1
                else:
                    failed_count += 1
                    failed_users.append(f"{username}: create_user rejected (duplicate username/email)")

            except Exception as e:
                failed_count += 1
                failed_users.append(f"{row.get('username', f'Row {index + 2}')}: {str(e)}")
                logger.error(f"Error importing user at row {index + 2}: {e}")

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
        vn = get_vn()
        training_data = vn.get_training_data()
        for _, row in training_data.iterrows():
            vn.remove_from_training(row["id"])
        st.toast("Training Data Deleted Successfully!")
    except Exception as e:
        st.error(f"An error occurred: {e}")
        logger.error(f"An error occurred: {e}")


def export_training_data_to_csv():
    try:
        vn = get_vn()
        training_data = vn.get_training_data()

        if isinstance(training_data, DataFrame) and not training_data.empty:
            csv_buffer = io.StringIO()
            training_data.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()

            import datetime

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_data_{timestamp}.csv"

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


def export_users_to_csv():
    try:
        try:
            admin_id_raw = st.session_state.cookies.get("user_id")
        except Exception:
            admin_id_raw = None
        if not admin_id_raw:
            st.error("Could not determine current user.")
            return
        admin_id = int(admin_id_raw)

        rows, n = get_users_for_export(admin_id)

        if n == 0:
            st.warning("No users available to export.")
            return

        df = DataFrame(rows)
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()

        import datetime

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"users_{timestamp}.csv"

        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name=filename,
            mime="text/csv",
            help="Download users as CSV file",
        )
    except Exception as e:
        st.error(f"An error occurred during CSV export: {e}")
        logger.error(f"User export error: {e}")


def import_training_data_from_csv(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)

        required_columns = ["training_data_type", "question", "content"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
            return

        vn = get_vn()
        success_count = 0
        error_count = 0

        progress_bar = st.progress(0)
        total_rows = len(df)

        for index, row in df.iterrows():
            try:
                progress_bar.progress((index + 1) / total_rows)

                if pd.isna(row["content"]) or str(row["content"]).strip() == "":
                    error_count += 1
                    continue

                training_type = str(row["training_data_type"]).lower()
                question = str(row["question"]) if not pd.isna(row["question"]) else None
                content = str(row["content"])

                if training_type == "sql" and question and question.strip():
                    vn.train(question=question, sql=content)
                elif training_type in ["ddl", "documentation"]:
                    vn.train(documentation=content)
                else:
                    vn.train(documentation=content)

                success_count += 1
            except Exception as row_error:
                error_count += 1
                logger.error(f"Error processing row {index}: {row_error}")

        progress_bar.empty()

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
        from utils.security_validator import security_validator

        if type == "sql":
            with st.form("add_training_data"):
                question = st.text_input("Question")
                content1 = st.text_input("Sql")
                if st.form_submit_button("Add"):
                    if question == "" or content1 == "":
                        st.error("Please fill all fields.")
                    else:
                        is_valid, violations = security_validator.validate_sql_content(content1)
                        if not is_valid:
                            st.error(f"SQL contains forbidden references: {', '.join(violations)}")
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
                        is_valid, violations = security_validator.validate_documentation(content2)
                        if not is_valid:
                            st.error(f"Documentation contains forbidden references: {', '.join(violations)}")
                        else:
                            get_vn().train(documentation=content2)
                            st.toast("Training Data Added Successfully!")
                            st.rerun()
    except Exception as e:
        st.error(f"An error occurred: {e}")
        logger.error(f"An error occurred: {e}")


@st.dialog("Confirm destructive action")
def confirm_destructive(body_md: str, token: str, on_confirm, *, button_label: str = "Confirm"):
    st.markdown(body_md)
    st.warning("This action cannot be undone.")
    typed = st.text_input(f"Type `{token}` to confirm:", key=f"_confirm_typed_{button_label}")
    cols = st.columns(2)
    with cols[0]:
        if st.button("Cancel", key=f"_confirm_cancel_{button_label}"):
            st.rerun()
    with cols[1]:
        if st.button(
            button_label,
            type="primary",
            disabled=(typed != token),
            key=f"_confirm_go_{button_label}",
        ):
            on_confirm()
            st.rerun()


def _create_user_dialog_body() -> None:
    """Body of the Create User dialog — extracted into a plain function
    so it can be unit-tested directly without going through ``st.dialog``
    (which requires a live Streamlit runtime). See
    ``tests/views/test_admin_users_validation.py``.

    Epic #179 enforces per-field inline error messaging for the three
    required fields (email, organization, role). Other failure modes
    (duplicate username/email, generic DB failure) keep the generic
    banner since the user has no per-field action.
    """
    roles = get_all_user_roles()
    role_id_by_name = {name: rid for rid, name, _ in roles}
    role_names = [name for rid, name, _ in roles]
    theme_options = user_selectable_themes()

    # Field errors from the previous submit attempt — keyed by field name
    # ("email", "organization", "role"). Rendered inline directly below
    # each input on the next render. We use session_state so the messages
    # survive the rerun that the form submit triggers; we clear stale
    # entries at the top of every render so an old error doesn't linger
    # on subsequent successful renders.
    errors_key = "create_user_dialog_field_errors"
    field_errors: dict[str, str] = st.session_state.get(errors_key, {})

    with st.form("create_user_dialog_form", clear_on_submit=False):
        cu_username = st.text_input("Username")
        cu_password = st.text_input("Temporary Password", type="password")
        cu_first = st.text_input("First Name")
        cu_last = st.text_input("Last Name")
        cu_email = st.text_input("Email")
        if "email" in field_errors:
            st.error(field_errors["email"])
        cu_organization = st.text_input("Organization")
        if "organization" in field_errors:
            st.error(field_errors["organization"])
        cu_role_name = st.selectbox(
            "Role", options=role_names, index=role_names.index("Patient") if "Patient" in role_names else 0
        )
        if "role" in field_errors:
            st.error(field_errors["role"])
        cu_theme = st.selectbox("Theme", options=theme_options, index=0)
        submitted = st.form_submit_button("Create User", type="primary")
        if submitted:
            # Pre-validate the non-Epic-179 fields (username, password,
            # first_name). The server-side validator only covers the
            # three required-by-#179 columns; username/password are still
            # required by other DB constraints, so block at the UI for a
            # better message.
            local_errors: dict[str, str] = {}
            if not cu_username.strip():
                local_errors["username"] = "Username is required."
            if not cu_password:
                local_errors["password"] = "Temporary password is required."
            if not cu_first.strip():
                local_errors["first_name"] = "First name is required."
            if local_errors:
                # Render these inline too via a banner since they're not
                # the per-field-error contract from #179 — keep the
                # interruption visible.
                for msg in local_errors.values():
                    st.error(msg)
                st.session_state[errors_key] = {}
                return

            try:
                ok = create_user(
                    cu_username,
                    cu_password,
                    cu_first,
                    cu_last,
                    role_id_by_name.get(cu_role_name),
                    email=cu_email,
                    organization=cu_organization,
                    theme=cu_theme,
                )
            except UserValidationError as ve:
                # Map each missing field to an inline per-field message.
                # Keys here MUST match the field-error rendering points
                # above ("email", "organization", "role"). See Epic #179
                # AC: "Form shows inline, per-field error messages for
                # missing values — not a generic save-failed banner."
                msgs: dict[str, str] = {}
                for field in ve.missing_fields:
                    if field == "email":
                        msgs["email"] = "A valid email address is required."
                    elif field == "organization":
                        msgs["organization"] = "Organization is required."
                    elif field == "role":
                        msgs["role"] = "A role must be selected."
                st.session_state[errors_key] = msgs
                st.rerun()
                return

            if ok:
                st.session_state[errors_key] = {}
                st.success("User created.")
                st.rerun()
            else:
                st.session_state[errors_key] = {}
                st.error("Failed to create user — username or email already exists.")


@st.dialog("Create User")
def create_user_dialog():
    _create_user_dialog_body()


@st.dialog("Bulk Import Users")
def bulk_import_dialog():
    st.info(
        "📁 Import users from Excel file: `./utils/config/user_list.xlsx`. "
        "Required columns: UserID, start_password, First Name, Last Name, Email, Organization."
    )
    bi_col1, bi_col2 = st.columns(2)
    with bi_col1:
        if st.button("Import Users", type="primary", key="bulk_import_dialog_run"):
            import_users()
    with bi_col2:
        if st.button("Close", key="bulk_import_dialog_close"):
            st.rerun()


@st.dialog("Export Users")
def export_users_dialog():
    st.info(
        "📤 Download all users as CSV. Columns match Bulk User Import "
        "(UserID, First Name, Last Name, Email, Organization, Role). "
        "Passwords are not included."
    )
    export_users_to_csv()
    if st.button("Close", key="export_users_dialog_close"):
        st.rerun()


@st.dialog("Set Password")
def set_password_dialog(selected: dict):
    st.markdown(f"Set a new password for **{selected['username']}**.")
    new_pw = st.text_input("New Password", type="password", key="set_pw_dialog_pw")
    sp_col1, sp_col2 = st.columns(2)
    with sp_col1:
        if st.button("Set Password", type="primary", key="set_pw_dialog_submit"):
            if not new_pw:
                st.error("Enter a new password.")
            elif admin_change_password(selected["id"], new_pw):
                st.success("Password updated.")
                st.rerun()
            else:
                st.error("Failed to update password.")
    with sp_col2:
        if st.button("Cancel", key="set_pw_dialog_cancel"):
            st.rerun()


def _delete_and_rerun(user_id: int):
    if delete_user(user_id):
        st.toast("User deleted.")
        st.rerun()
    else:
        st.error("Failed to delete user.")
