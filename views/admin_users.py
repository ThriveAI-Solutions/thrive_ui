"""Admin → Users sub-tab (Epic #144).

Relocated from views/user.py tab3. Post-Epic #129 layout: toolbar dialogs,
L/R split, 4 inner tabs (Profile / Preferences / Activity / Danger Zone).

Deep-link contracts:

- Outbound (this surface → Audit): The Activity inner tab's
  `View question audit for <username> →` button navigates to views/admin.py;
  the Admin Audit tab pre-filters to the target user via the existing
  `audit_trail_pref_user_id` session-state contract.

- Inbound (Audit Questions dialog → this surface, Feature #155): The
  Questions audit row-click dialog's "View user in Manage Users →" button
  sets `manage_users_pref_user_id` in session_state, then switches to
  views/admin.py. On the next render of this sub-tab, the pref is consumed
  (popped) and used to pre-populate `mu_selected_label` so the left-rail
  selectbox lands on the target user.
"""

import pandas as pd
import streamlit as st

from orm.functions import (
    get_all_user_roles,
    get_all_users,
    get_user_daily_stats,
    get_user_stats_for_all_users,
    update_user,
    update_user_preferences,
)
from orm.models import SessionLocal, User
from utils.enums import ThemeType, user_selectable_themes
from utils.quick_logger import get_logger

from views._admin_helpers import (
    bulk_import_dialog,
    confirm_destructive,
    create_user_dialog,
    export_users_dialog,
    set_password_dialog,
    _delete_and_rerun,
)

logger = get_logger(__name__)


def render(days_int: int | None = None) -> None:
    """Render the Manage Users admin sub-tab."""
    st.subheader("Manage Users")
    st.caption("Create, edit, or remove users. View settings and activity stats.")

    # Inbound deep-link from the Questions audit dialog (Feature #155):
    # consume `manage_users_pref_user_id`, look up the user, and pre-populate
    # `mu_selected_label` so the left-rail selectbox lands on the target user.
    # Only honour when the user hasn't already touched the selectbox this
    # session (mirrors the audit_trail_pref_user_id pattern in admin_analytics).
    pref_user_id = st.session_state.pop("manage_users_pref_user_id", None)
    if pref_user_id is not None and "mu_selected_label" not in st.session_state:
        try:
            all_users_for_pref = get_all_users()
            match = next((u for u in all_users_for_pref if u["id"] == int(pref_user_id)), None)
            if match:
                pref_label = f"{match['username']} ({match['first_name']} {match['last_name']}) - {match['role_name']}"
                st.session_state["mu_selected_label"] = pref_label
        except Exception as e:
            logger.warning("Manage Users deep-link prefill failed: %s", e)

    roles = get_all_user_roles()
    role_id_by_name = {name: rid for rid, name, _ in roles}
    role_names = [name for rid, name, _ in roles]

    stats_map = get_user_stats_for_all_users()

    # Toolbar — admin-action dialogs
    tb1, tb2, tb3_col, _spacer = st.columns([0.18, 0.18, 0.18, 0.46])
    with tb1:
        if st.button("+ Create User", type="primary", key="tb_create_user"):
            create_user_dialog()
    with tb2:
        if st.button("Import Users…", key="tb_import_users"):
            bulk_import_dialog()
    with tb3_col:
        if st.button("Export Users…", key="tb_export_users"):
            export_users_dialog()

    st.divider()

    left, right = st.columns([1, 2])
    with left:
        users = get_all_users()
        search = st.text_input("Search users", placeholder="Filter by username or name…", key="mu_search")
        if search:
            s = search.lower()
            users = [
                u for u in users if s in u["username"].lower() or s in f"{u['first_name']} {u['last_name']}`".lower()
            ]
        selected = None
        if users:
            user_options = {
                f"{u['username']} ({u['first_name']} {u['last_name']}) - {u['role_name']}": u for u in users
            }
            selected_label = st.selectbox("Select a user", options=list(user_options.keys()), key="mu_selected_label")
            selected = user_options[selected_label]
        else:
            st.info("No users found.")

    with right:
        if selected is None:
            st.info("Select a user from the left to view details.")
        else:
            st.markdown(
                f"**{selected['username']}** — {selected['first_name']} {selected['last_name']} "
                f"· {selected['role_name']}"
            )
            with SessionLocal() as session:
                db_user = session.query(User).filter(User.id == selected["id"]).first()
            prof_tab, prefs_tab, activity_tab, danger_tab = st.tabs(
                ["Profile", "Preferences", "Activity", "Danger Zone"]
            )

            with prof_tab:
                nu_username = st.text_input("Username", value=selected["username"])
                nu_first = st.text_input("First Name", value=selected["first_name"])
                nu_last = st.text_input("Last Name", value=selected["last_name"])
                nu_email = st.text_input("Email", value=selected.get("email") or "")
                nu_organization = st.text_input("Organization", value=selected.get("organization") or "")
                nu_role_name = st.selectbox(
                    "Role",
                    options=role_names,
                    index=role_names.index(selected["role_name"]) if selected["role_name"] in role_names else 0,
                )
                theme_options = user_selectable_themes()
                current_theme = selected.get("theme", ThemeType.HEALTHELINK.value)
                nu_theme = st.selectbox(
                    "Theme",
                    options=theme_options,
                    index=theme_options.index(current_theme) if current_theme in theme_options else 0,
                )
                prof_btn_cols = st.columns(2)
                with prof_btn_cols[0]:
                    if st.button("Save Profile", key="save_profile", type="primary"):
                        ok = update_user(
                            selected["id"],
                            nu_username,
                            nu_first,
                            nu_last,
                            role_id_by_name.get(nu_role_name),
                            nu_theme,
                            email=nu_email.strip() or None,
                            organization=nu_organization.strip() or None,
                        )
                        if ok:
                            st.success("Profile updated.")
                            st.rerun()
                        else:
                            st.error("Failed to update profile.")
                with prof_btn_cols[1]:
                    if st.button("Set Password…", key="set_password_btn"):
                        set_password_dialog(selected)

            with prefs_tab:
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
                else:
                    st.info("Preferences unavailable.")

            with activity_tab:
                s = stats_map.get(
                    selected["id"],
                    {"questions": 0, "charts": 0, "errors": 0, "dataframes": 0, "summaries": 0},
                )
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Questions", s["questions"])
                m2.metric("Charts", s["charts"])
                m3.metric("Errors", s["errors"])
                m4.metric("DataFrames", s["dataframes"])
                m5.metric("Summaries", s["summaries"])

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

                if st.button(
                    f"View question audit for {selected['username']} →",
                    type="primary",
                    key="activity_audit_deeplink_btn",
                ):
                    st.session_state["audit_trail_pref_user_id"] = selected["id"]
                    st.switch_page("views/admin.py")

            with danger_tab:
                if st.button("Delete User", type="primary", key="danger_delete_user_btn"):
                    confirm_destructive(
                        body_md=(
                            f"Permanently deletes user **{selected['username']}** "
                            f"({selected['first_name']} {selected['last_name']}, {selected['role_name']}). "
                            "Their account, preferences, and message history will be removed."
                        ),
                        token="DELETE",
                        on_confirm=lambda: _delete_and_rerun(selected["id"]),
                        button_label="Delete User",
                    )
