"""My Account page — Profile + Change Password (Epic #144 / #145).

Replaces the My Account tab that previously lived inside the merged
``User Settings`` page (views/user.py). The Training Data and Manage Users
admin tabs from the old User Settings page now live under the Admin umbrella
(views/admin.py).

Display preferences are intentionally not rendered here — they remain owned
by the sidebar ⚙️ Settings dialog introduced by Epic #140.
"""

import streamlit as st

from orm.functions import UserValidationError, change_password, update_user
from orm.models import RoleTypeEnum, SessionLocal, User
from utils.quick_logger import get_logger

logger = get_logger(__name__)

# Defensive guards mirror the patterns from the old views/user.py so the page
# stays importable when no Streamlit runtime is attached (tests).
try:
    user_id = st.session_state.cookies.get("user_id")
except Exception:
    user_id = None
try:
    user_role = st.session_state.get("user_role", RoleTypeEnum.PATIENT.value)
except Exception:
    user_role = RoleTypeEnum.PATIENT.value


st.title("My Account")

if user_id:
    try:
        with SessionLocal() as session:
            _me = session.query(User).filter(User.id == int(user_id)).first()
    except Exception:
        _me = None
    if _me is not None:
        st.markdown("**Profile**")
        # Per Epic #179 the My Account profile form must surface per-field
        # inline errors for missing/invalid email or organization rather
        # than a generic "save failed" banner.
        my_errors_key = f"my_account_field_errors_{user_id}"
        my_errors: dict[str, str] = st.session_state.get(my_errors_key, {})
        with st.form("my_profile_form"):
            my_email = st.text_input("Email", value=_me.email or "")
            if "email" in my_errors:
                st.error(my_errors["email"])
            my_organization = st.text_input("Organization", value=_me.organization or "")
            if "organization" in my_errors:
                st.error(my_errors["organization"])
            if st.form_submit_button("Save Profile", type="primary"):
                email_to_send = my_email.strip()
                org_to_send = my_organization.strip()
                try:
                    ok = update_user(
                        int(user_id),
                        email=email_to_send,
                        organization=org_to_send,
                    )
                except UserValidationError as ve:
                    msgs: dict[str, str] = {}
                    for field in ve.missing_fields:
                        if field == "email":
                            msgs["email"] = "A valid email address is required."
                        elif field == "organization":
                            msgs["organization"] = "Organization is required."
                    st.session_state[my_errors_key] = msgs
                    st.rerun()
                else:
                    if ok:
                        st.session_state[my_errors_key] = {}
                        st.success("Profile updated.")
                        st.rerun()
                    else:
                        st.session_state[my_errors_key] = {}
                        st.error("Failed to update profile — email may already be in use.")
        st.divider()

st.markdown("**Change Password**")
with st.form("change_password_form"):
    current_password = st.text_input("Current Password", type="password")
    new_password = st.text_input("New Password", type="password")
    confirm_new_password = st.text_input("Confirm New Password", type="password")
    submit_button = st.form_submit_button("Change Password", type="primary")

    if submit_button:
        if new_password != confirm_new_password:
            st.error("New password and confirmation do not match.")
        else:
            if change_password(user_id, current_password, new_password):
                st.success("Password changed successfully.")
            else:
                st.error("Current password is incorrect.")

st.divider()
st.caption("Display preferences live in the sidebar ⚙️ Settings dialog.")
