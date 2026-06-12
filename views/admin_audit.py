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
    # Per-rerun guard: Streamlit allows only one dialog open() per script run, but
    # st.tabs runs ALL tab bodies on every rerun and each tab has its own dataframe
    # selection state that persists across reruns. Whichever tab claims this flag
    # first in a rerun gets to open its dialog; the others skip. Reset at the top
    # so a new rerun starts with a clean slot.
    st.session_state["_audit_dialog_claimed_this_rerun"] = False

    # Audit-vs-external guard: admin.py renders Users -> Training -> Analytics ->
    # Audit -> Feedback in order. If an earlier tab already opened a dialog
    # (Export Users / Create User / Bulk Import / Set Password / confirm_destructive),
    # the dialog slot is gone. Calling a second @st.dialog raises and produces the
    # "external dialog flashes, audit dialog reappears" symptom. The Admin Actions
    # table is the worst case: Export Users writes a USER_EXPORT row via
    # log_admin_action, the next get_admin_actions_page reorders items, and the
    # data_editor's index-keyed sticky tick now points to the new top row, so the
    # per-tab open_id gate sees a mismatch and tries to fire. Mark the slot claimed
    # here so all three inner-tab gates skip.
    try:
        from streamlit.runtime.scriptrunner_utils.script_run_context import (
            get_script_run_ctx,
        )

        ctx = get_script_run_ctx()
        if ctx is not None and getattr(ctx, "has_dialog_opened", False):
            st.session_state["_audit_dialog_claimed_this_rerun"] = True
    except Exception:
        # If Streamlit moves these internals we just lose the extra guard and
        # revert to the audit-vs-audit-only behavior that existed before.
        pass

    inner = st.tabs(["Questions", "Admin Actions", "User Activity"])
    with inner[0]:
        _render_audit_trail_tab(days_int)
    with inner[1]:
        _render_audit_tab(days_int)
    with inner[2]:
        _render_activity_tab(days_int)
