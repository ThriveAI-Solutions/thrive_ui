"""Admin umbrella page (Epic #144 / #145).

Replaces the three former admin top-level pages (Admin Analytics, Feedback
Dashboard, Agentic Analytics) plus the admin-only sub-tabs of User Settings
(Training Data, Manage Users). Five sub-tabs in this order:

  Users · Training · Analytics · Audit · Feedback

Gated at the top of the script by _guard_admin(); each sub-tab's underlying
render function may also call _guard_admin() as defense in depth.

A single time-range segmented control at the top of the page drives the
days_int parameter passed down to each sub-tab.
"""

import streamlit as st

from views import (
    admin_analytics_consolidated,
    admin_audit,
    admin_feedback,
    admin_training,
    admin_users,
)
from views.admin_analytics import _guard_admin

_guard_admin()

st.title("Admin")

# Shared time-range control. Drives every sub-tab's days_int parameter.
control_cols = st.columns([3, 1])
with control_cols[0]:
    days = st.segmented_control(
        "Time Range",
        options=["7 days", "30 days", "90 days"],
        selection_mode="single",
        default="30 days",
        key="admin_time_range",
    )
with control_cols[1]:
    if st.button("Refresh Data", help="Clear cached data and reload metrics"):
        from views.admin_analytics import _read_metrics, _read_user_org_question_stats

        _read_metrics.clear()
        _read_user_org_question_stats.clear()
        from views.errors import _load as _errors_load, _load_aggregates as _errors_load_aggregates

        _errors_load.clear()
        _errors_load_aggregates.clear()
        st.rerun()

days_int = {"7 days": 7, "30 days": 30, "90 days": 90}.get(days or "30 days", 30)

tabs = st.tabs(["Users", "Training", "Analytics", "Audit", "Feedback"])
with tabs[0]:
    admin_users.render(days_int)
with tabs[1]:
    admin_training.render(days_int)
with tabs[2]:
    admin_analytics_consolidated.render(days_int)
with tabs[3]:
    admin_audit.render(days_int)
with tabs[4]:
    admin_feedback.render(days_int)
