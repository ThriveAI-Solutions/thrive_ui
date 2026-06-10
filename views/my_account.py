"""My Account page — Profile + Change Password (Epic #144 / #145).

Replaces the My Account tab that previously lived inside the merged
``User Settings`` page (views/user.py). The Training Data and Manage Users
admin tabs from the old User Settings page now live under the Admin umbrella
(views/admin.py).

Display preferences are intentionally not rendered here — they remain owned
by the sidebar ⚙️ Settings dialog introduced by Epic #140.
"""

import streamlit as st

from orm.functions import change_password, update_user
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
        with st.form("my_profile_form"):
            my_email = st.text_input("Email", value=_me.email or "")
            my_organization = st.text_input("Organization", value=_me.organization or "")
            if st.form_submit_button("Save Profile", type="primary"):
                ok = update_user(
                    int(user_id),
                    email=my_email.strip() or None,
                    organization=my_organization.strip() or None,
                )
                if ok:
                    st.success("Profile updated.")
                    st.rerun()
                else:
                    st.error(
                        "Failed to update profile. Possible causes: email already "
                        "in use, email format is invalid, or organization is empty."
                    )
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
