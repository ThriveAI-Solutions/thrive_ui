"""Admin → Audit sub-tab (Epic #144 / #148).

Consolidates three audit-flavored surfaces into one umbrella:
- Questions — the question audit trail introduced by Epic #133 / Feature #135
- Admin Actions — the admin action audit log (was Admin Analytics → Admin Audit)
- User Activity — the user activity log (was Admin Analytics → User Activity,
  minus the Activity Type pie chart cut by Epic #144 / #147)

All cross-Epic deep-link contracts (`audit_trail_pref_user_id` from #133,
the `View question audit for <username> →` button from Feature #136, and
the `View all in audit →` link from Feature #142) target this sub-tab via
st.switch_page("views/admin.py") — the outer Admin tab labelled "Audit" is
selected client-side by JS shim in the chat-input / Manage Users surfaces.
"""

import streamlit as st

from views.admin_analytics import _render_activity_tab, _render_audit_tab, _render_audit_trail_tab


def render(days_int: int) -> None:
    inner = st.tabs(["Questions", "Admin Actions", "User Activity"])
    with inner[0]:
        _render_audit_trail_tab(days_int)
    with inner[1]:
        _render_audit_tab(days_int)
    with inner[2]:
        _render_activity_tab(days_int)
