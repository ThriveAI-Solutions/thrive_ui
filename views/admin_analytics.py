"""Admin Analytics tabbed surface.

Two-ledger error-count contract (Epic #161):

  - **Ledger A — Chat-flow errors.** Counted from ``Message`` rows where
    ``Message.type == ERROR``. Consumed by the Admin Overview "Chat Errors"
    KPI (rendered in :func:`_render_overview_tab`). Surfaces the errors
    that interrupted a chat turn.

  - **Ledger B — System errors.** The 3-source union (``thrive_error_log``
    + agent-run failures + JSONL fallback file) rolled up by
    :func:`orm.error_read_model.query_aggregates` and cached via
    :func:`views.errors._load_aggregates`. Consumed by the Errors tab and
    by the Admin Overview "Critical System Errors" KPI. The Overview KPI
    shares the ``_load_aggregates`` cache so it always equals the Errors
    tab's Critical card under the same ``days_int``.

New error-count surfaces MUST declare which ledger they consume and route
through the corresponding loader. Do not introduce a third ad-hoc counter
without updating this contract — that is the drift Epic #161 exists to
prevent.
"""

import datetime as dt
import json

import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import case, func

from orm.models import Message, RoleTypeEnum, SessionLocal, User
from utils.enums import MessageType, RoleType
from utils.quick_logger import get_logger

logger = get_logger(__name__)

# Cache TTL for analytics queries (5 minutes)
ANALYTICS_CACHE_TTL_SECONDS = 300

# Epic #169 / Feature #170: shared copy for the labeled View checkbox column
# used on all three audit tables (Questions, Admin Actions, User Activity).
# Keeping these as module-level constants ensures the three sites stay
# literally identical and gives the tests a single string to assert against.
# Interaction model: auto-open dialog when exactly one row is ticked (no
# action button) — re-ticking the same row after closing is supported via
# the per-tab ``open_id`` session-state gate cleared on zero-tick reruns.
_VIEW_COLUMN_LABEL = "View"
_VIEW_COLUMN_HELP = "Tick a box to open the detail dialog"


def _kpi_card(label: str, value, help_text: str | None = None):
    """Render a KPI card with label, value, and optional help text."""
    c = st.container(border=True)
    with c:
        st.markdown(f"**{label}**")
        st.markdown(f"<h3 style='margin-top:0'>{value}</h3>", unsafe_allow_html=True)
        if help_text:
            st.caption(help_text)


def _guard_admin():
    """Ensure only admin users can access this page."""
    if st.session_state.get("user_role") != RoleTypeEnum.ADMIN.value:
        st.error("You don't have permission to view this page.")
        st.stop()


@st.cache_data(ttl=ANALYTICS_CACHE_TTL_SECONDS, show_spinner="Loading metrics...")
def _read_metrics(days: int = 30):
    """Read message-based metrics for the Overview tab.

    Results are cached for 5 minutes to improve dashboard responsiveness.
    """
    chart_types = [
        MessageType.PLOTLY_CHART.value,
        MessageType.ST_LINE_CHART.value,
        MessageType.ST_BAR_CHART.value,
        MessageType.ST_AREA_CHART.value,
        MessageType.ST_SCATTER_CHART.value,
    ]

    since = dt.datetime.now() - dt.timedelta(days=days)

    with SessionLocal() as session:
        users_count = session.query(func.count()).select_from(User).scalar() or 0
        active_users = (
            session.query(func.count(func.distinct(Message.user_id)))
            .filter(Message.role == RoleType.USER.value, Message.created_at >= since)
            .scalar()
            or 0
        )

        # Over-time counts
        date_expr = func.strftime("%Y-%m-%d", Message.created_at)
        over_time = (
            session.query(
                date_expr.label("d"),
                func.sum(case((Message.role == RoleType.USER.value, 1), else_=0)).label("questions"),
                func.sum(case((Message.type.in_(chart_types), 1), else_=0)).label("charts"),
                func.sum(case((Message.type == MessageType.SUMMARY.value, 1), else_=0)).label("summaries"),
                func.sum(case((Message.type == MessageType.SQL.value, 1), else_=0)).label("sql"),
                func.sum(case((Message.type == MessageType.ERROR.value, 1), else_=0)).label("errors"),
            )
            .filter(Message.created_at >= since)
            .group_by("d")
            .order_by("d")
            .all()
        )

        # Over-time conversion: results (charts+dataframes+summaries) / questions
        result_over_time = (
            session.query(
                date_expr.label("d"),
                func.sum(case((Message.role == RoleType.USER.value, 1), else_=0)).label("questions"),
                func.sum(
                    case(
                        (
                            Message.type.in_([MessageType.DATAFRAME.value, MessageType.SUMMARY.value] + chart_types),
                            1,
                        ),
                        else_=0,
                    )
                ).label("results"),
            )
            .filter(Message.created_at >= since)
            .group_by("d")
            .order_by("d")
            .all()
        )

        # Elapsed stats (seconds) for assistant messages
        # Pull raw elapsed times and compute stats in Python for SQLite compatibility
        def _compute_stats_for_query(q):
            import numpy as np

            vals = [float(v or 0.0) for (v,) in q if v is not None]
            n = len(vals)
            if n == 0:
                return {"avg": 0.0, "min": 0.0, "max": 0.0, "stddev": 0.0, "median": 0.0, "n": 0}
            arr = np.array(vals, dtype=float)
            return {
                "avg": float(arr.mean()),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "stddev": float(arr.std(ddof=0)),
                "median": float(np.median(arr)),
                "n": int(n),
            }

        overall_q = (
            session.query(Message.elapsed_time)
            .filter(Message.role == RoleType.ASSISTANT.value, Message.created_at >= since)
            .all()
        )
        overall_stats = _compute_stats_for_query(overall_q)

        def _perf_for_type(msg_type: str):
            q = (
                session.query(Message.elapsed_time)
                .filter(
                    Message.role == RoleType.ASSISTANT.value,
                    Message.type == msg_type,
                    Message.created_at >= since,
                )
                .all()
            )
            return _compute_stats_for_query(q)

        perf_types = {
            "sql": _perf_for_type(MessageType.SQL.value),
            "summary": _perf_for_type(MessageType.SUMMARY.value),
            "chart": _perf_for_type(MessageType.PLOTLY_CHART.value),
            "dataframe": _perf_for_type(MessageType.DATAFRAME.value),
        }

        return (
            users_count,
            active_users,
            over_time,
            result_over_time,
            overall_stats,
            perf_types,
        )


def _to_dense_days(rows, days: int):
    """Convert sparse date rows to dense daily series."""
    today = dt.date.today()
    start = today - dt.timedelta(days=days - 1)
    by_date = {r.d: r for r in rows}
    out = []
    for i in range(days):
        d = start + dt.timedelta(days=i)
        k = d.strftime("%Y-%m-%d")
        r = by_date.get(k)
        out.append(
            {
                "date": k,
                "questions": int((r.questions if r else 0) or 0),
                "charts": int((r.charts if r else 0) or 0),
                "summaries": int((r.summaries if r else 0) or 0),
                "sql": int((r.sql if r else 0) or 0),
                "errors": int((r.errors if r else 0) or 0),
            }
        )
    return pd.DataFrame(out)


# ============== Tab Rendering Functions ==============


def _render_overview_tab(days_int: int):
    """Render the Overview tab (existing content + new KPIs)."""
    from orm.logging_functions import get_activity_stats, get_llm_stats
    from views.errors import _load_aggregates as _errors_load_aggregates

    (
        users_count,
        active_users,
        over_time,
        result_over_time,
        overall_stats,
        perf_types,
    ) = _read_metrics(days=days_int)

    # Get stats from new logging tables
    llm_stats = get_llm_stats(days=days_int)
    activity_stats = get_activity_stats(days=days_int)
    # Ledger B (System errors) — shared cache with the Errors tab.
    # Importing _load_aggregates here mirrors the precedent at the global
    # Refresh Data button below and guarantees Overview's "Critical System
    # Errors" KPI always equals the Errors tab's Critical card under the
    # same days_int (Epic #161).
    error_aggregates = _errors_load_aggregates(days_int)

    # KPIs row - original + new
    k1, k2, k3, k4, k5, k6, k7, k8 = st.columns(8)
    df = _to_dense_days(over_time, days_int)
    with k1:
        _kpi_card("Users", users_count)
    with k2:
        _kpi_card("Questions", int(df["questions"].sum()))
    with k3:
        _kpi_card(
            "Chat Errors",
            int(df["errors"].sum()),
            help_text=("Chat-flow errors (Message.type=ERROR). Distinct from the Errors tab's System Errors ledger."),
        )
    with k4:
        _kpi_card("Active Users", active_users)
    with k5:
        _kpi_card("LLM Queries", llm_stats["total"], "SQL generations")
    with k6:
        _kpi_card("Avg Latency", f"{llm_stats['avg_latency_ms']:.0f}ms")
    with k7:
        _kpi_card("Logins Today", activity_stats["logins_today"])
    with k8:
        _kpi_card(
            "Critical System Errors",
            error_aggregates["critical"],
            help_text=(
                "Severity=critical across ErrorLog + Agent Runs + Fallback "
                "File. Matches the Errors tab's Critical card."
            ),
        )

    st.divider()

    # Over time chart
    if not df.empty:
        mdf = df.melt(
            id_vars=["date"],
            value_vars=["questions", "sql", "summaries", "charts", "errors"],
            var_name="metric",
            value_name="count",
        )
        mdf["date"] = pd.to_datetime(mdf["date"])
        fig = px.line(mdf, x="date", y="count", color="metric", markers=True)
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), legend_title_text="")
        st.plotly_chart(fig, width="stretch")

        # Conversion over time
        start = (pd.Timestamp.today().normalize() - pd.Timedelta(days=days_int - 1)).date()
        by_date = {r.d: {"questions": int(r.questions or 0), "results": int(r.results or 0)} for r in result_over_time}
        conv_rows = []
        for i in range(days_int):
            d = (pd.Timestamp(start) + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
            vals = by_date.get(d, {"questions": 0, "results": 0})
            q = vals["questions"]
            res = vals["results"]
            conv_rows.append({"date": d, "questions": q, "results": res, "conversion": (res / q) if q else 0})
        conv_df = pd.DataFrame(conv_rows)
        conv_df["date"] = pd.to_datetime(conv_df["date"])
        cfig = px.line(conv_df, x="date", y="conversion", title="Conversion (results/questions) over time")
        cfig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(cfig, width="stretch")

    st.divider()

    # Elapsed stats
    st.subheader("Performance (Assistant elapsed time)")
    avg = float(overall_stats.get("avg", 0))
    _min = float(overall_stats.get("min", 0))
    _max = float(overall_stats.get("max", 0))
    stddev = float(overall_stats.get("stddev", 0))
    median = float(overall_stats.get("median", 0))
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        _kpi_card("Avg (s)", round(avg, 3))
    with c2:
        _kpi_card("Min (s)", round(_min, 3))
    with c3:
        _kpi_card("Max (s)", round(_max, 3))
    with c4:
        _kpi_card("Std Dev (s)", round(stddev, 3))
    with c5:
        _kpi_card("Median (s)", round(median, 3))

    # Per-type stats
    st.subheader("Performance by Output Type")
    pt_df = pd.DataFrame(
        [
            {
                "Type": t.title(),
                "Avg (s)": round(perf_types[t]["avg"], 3),
                "Min (s)": round(perf_types[t]["min"], 3),
                "Max (s)": round(perf_types[t]["max"], 3),
                "Std Dev (s)": round(perf_types[t]["stddev"], 3),
                "Median (s)": round(perf_types[t]["median"], 3),
                "Samples": perf_types[t]["n"],
            }
            for t in ["sql", "summary", "chart", "dataframe"]
        ]
    )
    st.dataframe(pt_df, width="stretch", hide_index=True)

    # Distribution
    with SessionLocal() as session:
        dist = (
            session.query(func.coalesce(Message.elapsed_time, 0).label("elapsed"))
            .filter(Message.role == RoleType.ASSISTANT.value)
            .all()
        )
    if dist:
        import numpy as np

        arr = np.array([float(r.elapsed or 0) for r in dist])
        hist = pd.DataFrame({"elapsed": arr})
        fig2 = px.histogram(hist, x="elapsed", nbins=40, title="Elapsed time distribution (s)")
        fig2.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig2, width="stretch")

    st.divider()

    # Latest Questions — compact preview that links to the canonical Audit Trail tab (#136).
    st.subheader("Latest Questions")
    from orm.logging_functions import get_question_audit_page

    preview = get_question_audit_page(
        {"usernames": [], "orgs": [], "days": days_int, "search": None},
        page=1,
        page_size=10,
    )
    preview_items = preview.get("items", [])
    if preview_items:
        compact_rows = [
            {
                "User": it["username"],
                "Question": _truncate(it["question"], 120),
                "Asked At": it["asked_at"],
                "Status": _AUDIT_STATUS_EMOJI.get(it["status"], it["status"]),
            }
            for it in preview_items
        ]
        st.dataframe(pd.DataFrame(compact_rows), width="stretch", hide_index=True)
    else:
        st.info("No questions in the selected time range.")
    # Streamlit's st.tabs widget has no server-side API to change the active
    # tab and st.switch_page is a no-op on the same page, so we use a small
    # JS shim served via components.v1.html that clicks the parent window's
    # Audit Trail tab on link click.
    st.components.v1.html(
        """
        <a href="#" id="goto-audit-link" onclick="
          (function(){
            try {
              var tabs = window.parent.document.querySelectorAll('button[role=tab]');
              for (var i = 0; i < tabs.length; i++) {
                if (tabs[i].textContent.trim() === 'Audit') { tabs[i].click(); break; }
              }
            } catch (e) { console.error('audit-tab nav failed', e); }
            return false;
          })();
          return false;
        " style="display:inline-block;padding:0.5rem 1rem;background:#0b5258;
                 color:white;text-decoration:none;border-radius:0.5rem;
                 font-weight:600;font-family:inherit;">
          View full Audit Trail →
        </a>
        """,
        height=60,
    )


@st.cache_data(ttl=ANALYTICS_CACHE_TTL_SECONDS, show_spinner=False)
def _cached_audit_filter_options(days_int: int) -> dict:
    from orm.logging_functions import get_question_audit_filter_options

    return get_question_audit_filter_options(days=days_int)


_AUDIT_STATUS_EMOJI = {"Success": "✅ Success", "Error": "❌ Error", "Empty": "⚪ Empty"}


def _truncate(text, length):
    if text is None:
        return ""
    text = str(text)
    return text if len(text) <= length else text[: length - 1] + "…"


def _render_audit_question_dialog_body(item: dict) -> None:
    """Render the body of the Questions audit dialog (#155).

    Split out from the `@st.dialog`-decorated wrapper so tests can call the
    body directly with a stubbed `st` module. The decorated wrapper
    `_render_audit_question_dialog` is what production calls.
    """
    from io import StringIO

    from agent.observability_gate import role_can_see_query_details

    current_role = st.session_state.get("user_role")
    can_see_query_details = role_can_see_query_details(current_role)

    # Header line
    org = item.get("organization") or "(no org)"
    st.markdown(f"**{item['asked_at']}** · {item['username']} · {org}")

    # Full question
    st.markdown("**Question**")
    st.write(item.get("question") or "_(empty)_")

    # Role-gated: Generated SQL with built-in copy-to-clipboard
    if can_see_query_details:
        st.markdown("**Generated SQL**")
        st.code(item.get("sql_text") or "(no SQL)", language="sql")

    # Summary
    st.markdown("**Summary**")
    summary = item.get("summary_text")
    st.write(summary if summary else "_(no summary)_")

    # Role-gated: full DataFrame (no .head(5) — dataframe_preview holds the full JSON)
    df_preview = item.get("dataframe_preview")
    if df_preview and can_see_query_details:
        st.markdown("**Result DataFrame**")
        try:
            df_obj = pd.read_json(StringIO(df_preview))
            st.dataframe(df_obj, width="stretch", hide_index=True)
        except Exception:
            try:
                df_obj = pd.read_json(df_preview)
                st.dataframe(df_obj, width="stretch", hide_index=True)
            except Exception:
                st.text(str(df_preview)[:1000])

    # Error block when present
    err = item.get("error_text")
    if err:
        st.markdown("**Error**")
        st.code(err, language="text")

    # Metadata caption
    st.caption(
        f"Message ID {item.get('user_message_id')} · Elapsed {round(float(item.get('elapsed_seconds') or 0.0), 3)}s"
    )

    # Outbound deep-link to Manage Users
    st.divider()
    if st.button(
        "View user in Manage Users →",
        key=f"audit_dialog_goto_users_{item.get('user_message_id')}",
        type="primary",
    ):
        st.session_state["manage_users_pref_user_id"] = item["user_id"]
        # JS tab-selection shim — mirrors the Audit-tab shim above but targets "Users".
        # Streamlit's st.tabs has no server-side API to change the active tab and
        # st.switch_page is a no-op on the same page, so we click the parent doc's
        # "Users" tab button after the rerun lands.
        st.components.v1.html(
            """
            <script>
              (function(){
                try {
                  var tabs = window.parent.document.querySelectorAll('button[role=tab]');
                  for (var i = 0; i < tabs.length; i++) {
                    if (tabs[i].textContent.trim() === 'Users') { tabs[i].click(); break; }
                  }
                } catch (e) { console.error('users-tab nav failed', e); }
              })();
            </script>
            """,
            height=0,
        )
        st.switch_page("views/admin.py")


@st.dialog("Question Audit Detail")
def _render_audit_question_dialog(item: dict) -> None:
    """`@st.dialog`-decorated wrapper invoked from `_render_audit_trail_tab`.

    Epic #169 / Feature #170 swapped the trigger primitive from
    ``st.dataframe(on_select=..., selection_mode='single-row')`` to a
    ``st.data_editor`` + labeled ``CheckboxColumn`` that auto-opens this
    dialog when exactly one row is ticked. Subsystems #156 and #157
    adopt the same pattern. See Epic #169 Architecture Considerations.
    """
    _render_audit_question_dialog_body(item)


def _render_audit_trail_tab(days_int: int):
    """Render the Question Audit Trail tab (#135)."""
    import datetime as _dt

    from orm.logging_functions import (
        get_question_audit_export,
        get_question_audit_page,
    )
    from orm.functions import get_all_users

    # [agent_logging].mode = "disabled" defense-in-depth.
    # Under "disabled" mode no audit rows are written, but we belt-and-suspenders
    # the trigger primitive so it cannot open the dialog.
    try:
        logging_mode = st.secrets.get("agent_logging", {}).get("mode", "full")
    except Exception:
        logging_mode = "full"
    selection_enabled = logging_mode != "disabled"

    options = _cached_audit_filter_options(days_int)

    # Honour deep-link pre-filter from Manage Users -> Activity
    pref_user_id = st.session_state.pop("audit_trail_pref_user_id", None)
    if pref_user_id is not None and "audit_user_filter" not in st.session_state:
        try:
            all_users = get_all_users()
            match = next((u for u in all_users if u["id"] == int(pref_user_id)), None)
            if match and match["username"] in options["usernames"]:
                st.session_state["audit_user_filter"] = [match["username"]]
        except Exception as e:
            logger.warning("Audit deep-link prefill failed: %s", e)

    # Scope options are sourced from the backend so the chip and the SQL
    # CASE stay in sync. Importing inside the function mirrors the existing
    # local-import convention in this module.
    from orm.logging_functions import ALL_SCOPES as _AUDIT_SCOPE_OPTIONS

    f1, f2, f3, f4 = st.columns([0.24, 0.24, 0.22, 0.30])
    with f1:
        usernames_sel = st.multiselect("User", options=options["usernames"], key="audit_user_filter")
    with f2:
        orgs_sel = st.multiselect("Organization", options=options["orgs"], key="audit_org_filter")
    with f3:
        scopes_sel = st.multiselect(
            "Scope",
            options=list(_AUDIT_SCOPE_OPTIONS),
            key="audit_scope_filter",
            help="Patient = single-patient questions (PHI access). "
            "Pop Health = cohort searches. Other = codes / KB / pure SQL. "
            "Legacy/Unknown = pre-agentic-replatform rows.",
        )
    with f4:
        search = st.text_input(
            "Search question or SQL",
            placeholder="Substring match (case-insensitive)",
            key="audit_search",
        )

    filters = {
        "usernames": usernames_sel,
        "orgs": orgs_sel,
        "days": int(days_int),
        "search": search.strip() if search else None,
        # ``set`` is JSON-unfriendly; backend accepts any iterable.
        "scopes": sorted(scopes_sel) if scopes_sel else None,
    }

    # Reset pagination on filter change
    filter_signature = json.dumps({**filters, "_days": days_int}, sort_keys=True, default=str)
    if st.session_state.get("audit_filter_signature") != filter_signature:
        st.session_state["audit_filter_signature"] = filter_signature
        st.session_state["audit_page_num"] = 1

    # Pagination controls (top row)
    if "audit_page_bump" in st.session_state:
        st.session_state["audit_page_num"] = max(
            1, int(st.session_state.get("audit_page_num", 1)) + int(st.session_state["audit_page_bump"])
        )
        del st.session_state["audit_page_bump"]
    if "audit_page_num" not in st.session_state:
        st.session_state["audit_page_num"] = 1

    pc1, pc2, _spacer = st.columns([1, 1, 6])
    with pc1:
        page_size = st.selectbox("Page size", options=[25, 50, 100, 200], index=1, key="audit_page_size")
    with pc2:
        st.number_input("Page", min_value=1, step=1, key="audit_page_num")
        page = int(st.session_state["audit_page_num"])

    result = get_question_audit_page(filters, page=page, page_size=int(page_size))
    items = result.get("items", [])
    total = int(result.get("total", 0))
    total_pages = max(1, (total + int(page_size) - 1) // int(page_size))

    # Table — Epic #169 / Feature #170 swaps the prior single-row selection
    # trigger (``st.dataframe(on_select=..., selection_mode='single-row')``)
    # for a manual ``st.data_editor`` + leading labeled ``CheckboxColumn``
    # that auto-opens the detail dialog on tick. Streamlit's row-selection
    # widget can't be labeled via ``column_config`` because it's a frontend
    # element, not a data column. A manual boolean data column gives us a
    # labeled header + help tooltip — the discoverability win the Epic
    # exists to deliver — while preserving the original one-step
    # ``selection_mode='single-row'`` interaction model.
    if items:
        table_rows = []
        for it in items:
            table_rows.append(
                {
                    # ``View`` is the leading checkbox column; only this one
                    # column is editable, all others are ``disabled=True``.
                    _VIEW_COLUMN_LABEL: False,
                    "Asked At": it["asked_at"],
                    "User": it["username"],
                    "Organization": it.get("organization") or "(no org)",
                    "Question": _truncate(it["question"], 120),
                    "SQL": _truncate(it.get("sql_text"), 80),
                    "Status": _AUDIT_STATUS_EMOJI.get(it["status"], it["status"]),
                    # Epic #166 / Feature #167: derived scope label —
                    # Patient / Pop Health / Other / Legacy/Unknown.
                    "Scope": it.get("scope") or "Legacy/Unknown",
                    "Elapsed (s)": round(float(it.get("elapsed_seconds") or 0.0), 3),
                }
            )

        if selection_enabled:
            edited_df = st.data_editor(
                pd.DataFrame(table_rows),
                width="stretch",
                hide_index=True,
                num_rows="fixed",
                key="audit_dataframe",
                column_config={
                    _VIEW_COLUMN_LABEL: st.column_config.CheckboxColumn(
                        _VIEW_COLUMN_LABEL,
                        help=_VIEW_COLUMN_HELP,
                        default=False,
                        width="small",
                    ),
                    "Asked At": st.column_config.TextColumn("Asked At", disabled=True),
                    "User": st.column_config.TextColumn("User", disabled=True),
                    "Organization": st.column_config.TextColumn("Organization", disabled=True),
                    "Question": st.column_config.TextColumn("Question", disabled=True),
                    "SQL": st.column_config.TextColumn("SQL", disabled=True),
                    "Status": st.column_config.TextColumn("Status", disabled=True),
                    "Scope": st.column_config.TextColumn("Scope", disabled=True),
                    "Elapsed (s)": st.column_config.TextColumn("Elapsed (s)", disabled=True),
                },
            )

            try:
                current_checked = [int(i) for i in edited_df.index[edited_df[_VIEW_COLUMN_LABEL]].tolist()]
            except Exception:
                current_checked = []

            # Single-select enforcement. The ``CheckboxColumn`` natively
            # allows multi-tick; we want radio-button semantics with
            # checkbox aesthetics. When the user ticks a new row while
            # another is already ticked, untick the older one(s) by
            # mutating ``data_editor``'s widget state at
            # ``st.session_state["audit_dataframe"]["edited_rows"]`` and
            # ``st.rerun()`` so the next render reflects the single tick.
            prev_checked = list(st.session_state.get("audit_questions_prev_view_checks", []))
            if len(current_checked) > 1:
                newly_checked = [i for i in current_checked if i not in prev_checked]
                keep_idx = newly_checked[0] if newly_checked else current_checked[0]
                editor_state = st.session_state.get("audit_dataframe")
                if isinstance(editor_state, dict):
                    edited_rows = editor_state.setdefault("edited_rows", {})
                    for idx in current_checked:
                        if idx == keep_idx:
                            continue
                        row_edits = edited_rows.get(idx)
                        if isinstance(row_edits, dict) and _VIEW_COLUMN_LABEL in row_edits:
                            del row_edits[_VIEW_COLUMN_LABEL]
                            if not row_edits:
                                del edited_rows[idx]
                st.session_state["audit_questions_prev_view_checks"] = [keep_idx]
                st.rerun()
            else:
                st.session_state["audit_questions_prev_view_checks"] = current_checked

            checked_count = len(current_checked)
            if checked_count == 1:
                row_idx = current_checked[0]
                if 0 <= row_idx < len(items):
                    selected_item = items[row_idx]
                    # Auto-open: dialog fires when exactly one row is
                    # ticked. Gate on the tab-specific ``open_id`` session
                    # key so the dialog doesn't re-fire on every rerun
                    # while the checkbox stays ticked. Cross-tab claim
                    # guard from PR #168 still applies as
                    # defense-in-depth.
                    open_id = st.session_state.get("audit_dialog_open_user_message_id")
                    if open_id != selected_item["user_message_id"]:
                        if not st.session_state.get("_audit_dialog_claimed_this_rerun"):
                            st.session_state["_audit_dialog_claimed_this_rerun"] = True
                            st.session_state["audit_dialog_open_user_message_id"] = selected_item["user_message_id"]
                            _render_audit_question_dialog(selected_item)
            elif checked_count == 0:
                # Allow re-ticking the same row to reopen the dialog by
                # clearing the gate.
                st.session_state.pop("audit_dialog_open_user_message_id", None)
        else:
            # agent_logging.mode == "disabled": render the table read-only;
            # no editor and no action button — the dialog is unreachable.
            # Drop the ``View`` column entry from each row so the read-only
            # display doesn't carry a meaningless checkbox column.
            readonly_rows = [{k: v for k, v in row.items() if k != _VIEW_COLUMN_LABEL} for row in table_rows]
            st.dataframe(
                pd.DataFrame(readonly_rows),
                width="stretch",
                hide_index=True,
                key="audit_dataframe",
            )
    else:
        st.info("No audit rows match the current filters.")

    # Pagination caption + Prev/Next
    cprev, cinfo, cnext = st.columns([1, 3, 1])
    with cprev:
        st.button(
            "Prev",
            disabled=int(page) <= 1,
            key="audit_prev_btn",
            on_click=lambda: st.session_state.update({"audit_page_bump": -1}),
        )
    with cinfo:
        st.caption(f"Page {int(page)} of {total_pages} • {total} total")
    with cnext:
        st.button(
            "Next",
            disabled=int(page) >= total_pages,
            key="audit_next_btn",
            on_click=lambda: st.session_state.update({"audit_page_bump": 1}),
        )

    # Export CSV
    if st.button("📥 Export filtered audit to CSV", key="audit_export_btn"):
        try:
            export_rows = get_question_audit_export(filters)
            if not export_rows:
                st.warning("No audit rows to export with current filters.")
            else:
                export_df = pd.DataFrame(
                    [
                        {
                            "Asked At": r["asked_at"],
                            "User": r["username"],
                            "Organization": r.get("organization") or "(no org)",
                            "Question": r["question"],
                            "SQL": r.get("sql_text") or "",
                            "Status": r["status"],
                            # Epic #166 / Feature #167: same derived label
                            # the grid surfaces. CSV must obey the Scope
                            # filter (transparent — same filters dict) AND
                            # carry the classification through to the file.
                            "Scope": r.get("scope") or "Legacy/Unknown",
                            "Elapsed (s)": round(float(r.get("elapsed_seconds") or 0.0), 3),
                        }
                        for r in export_rows
                    ]
                )
                ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    "Download question_audit_*.csv",
                    data=export_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"question_audit_{ts}.csv",
                    mime="text/csv",
                    key="audit_export_download",
                )
        except Exception as e:
            st.error(f"Export failed: {e}")
            logger.error("Audit export failed: %s", e)


def _render_llm_tab(days_int: int):
    """Render the LLM Performance tab."""
    from orm.logging_functions import (
        get_llm_over_time,
        get_llm_provider_breakdown,
        get_llm_stats,
        get_recent_llm_queries,
    )

    stats = get_llm_stats(days=days_int)

    # KPI Row
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        _kpi_card("Total Queries", stats["total"])
    with k2:
        _kpi_card("Avg Latency", f"{stats['avg_latency_ms']:.0f}ms")
    with k3:
        _kpi_card("Avg Tokens", f"{stats['avg_tokens']:.0f}")
    with k4:
        avg_rag = stats["avg_ddl_count"] + stats["avg_doc_count"] + stats["avg_example_count"]
        _kpi_card("Avg RAG Items", f"{avg_rag:.1f}")
    with k5:
        # Success rate placeholder - would need error correlation
        _kpi_card(
            "DDL/Doc/Ex", f"{stats['avg_ddl_count']:.1f}/{stats['avg_doc_count']:.1f}/{stats['avg_example_count']:.1f}"
        )

    st.divider()

    # Provider/Model Breakdown
    st.subheader("Provider & Model Breakdown")
    provider_data = get_llm_provider_breakdown(days=days_int)
    if provider_data:
        provider_df = pd.DataFrame(provider_data)
        col1, col2 = st.columns(2)
        with col1:
            # Provider bar chart
            provider_agg = provider_df.groupby("provider")["count"].sum().reset_index()
            fig = px.bar(provider_agg, x="count", y="provider", orientation="h", title="Queries by Provider")
            fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, width="stretch")
        with col2:
            # Model table
            st.dataframe(
                provider_df[["provider", "model", "count", "avg_latency"]].rename(
                    columns={"count": "Queries", "avg_latency": "Avg Latency (ms)"}
                ),
                width="stretch",
                hide_index=True,
            )
    else:
        st.info("No LLM query data available yet.")

    st.divider()

    # Latency Over Time
    st.subheader("Latency Over Time")
    over_time = get_llm_over_time(days=days_int)
    if over_time:
        ot_df = pd.DataFrame(over_time)
        ot_df["date"] = pd.to_datetime(ot_df["date"])
        fig = px.line(ot_df, x="date", y="avg_latency", markers=True, title="Average Latency (ms) Over Time")
        fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, width="stretch")

        # Query volume
        fig2 = px.bar(ot_df, x="date", y="queries", title="Query Volume Over Time")
        fig2.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig2, width="stretch")
    else:
        st.info("No LLM time series data available yet.")

    st.divider()

    # Recent Queries Drill-down
    st.subheader("Recent LLM Queries")
    recent = get_recent_llm_queries(days=days_int, limit=50)
    if recent:
        for item in recent:
            with st.expander(
                f"**{item['question'][:80]}...** - {item['provider']}/{item['model']} - {item['total_time_ms'] or 0}ms",
                expanded=False,
            ):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Latency", f"{item['total_time_ms'] or 0}ms")
                with col2:
                    st.metric("Tokens", item["total_tokens"] or "N/A")
                with col3:
                    st.metric(
                        "RAG Items",
                        f"{item['ddl_count'] or 0}D/{item['doc_count'] or 0}Doc/{item['example_count'] or 0}Ex",
                    )

                st.markdown("**Question:**")
                st.info(item["question"])

                if item["generated_sql"]:
                    st.markdown("**Generated SQL:**")
                    st.code(item["generated_sql"], language="sql")

                # Show RAG context if available
                if item["ddl_statements"]:
                    with st.expander(f"DDL Statements ({item['ddl_count'] or 0} items)", expanded=False):
                        try:
                            ddls = json.loads(item["ddl_statements"])
                            for i, ddl in enumerate(ddls):
                                st.markdown(f"**DDL #{i + 1}:**")
                                st.code(ddl, language="sql")
                        except Exception:
                            st.code(item["ddl_statements"], language="sql")

                if item["documentation_snippets"]:
                    with st.expander(f"Documentation Snippets ({item['doc_count'] or 0} items)", expanded=False):
                        try:
                            docs = json.loads(item["documentation_snippets"])
                            for i, doc in enumerate(docs):
                                st.markdown(f"**Documentation #{i + 1}:**")
                                st.markdown(doc)
                                if i < len(docs) - 1:
                                    st.divider()
                        except Exception:
                            st.markdown(item["documentation_snippets"])

                if item.get("question_sql_examples"):
                    with st.expander(f"Q&A Examples ({item['example_count'] or 0} items)", expanded=False):
                        try:
                            examples = json.loads(item["question_sql_examples"])
                            for i, ex in enumerate(examples):
                                st.markdown(f"**Example #{i + 1}:**")
                                if isinstance(ex, dict):
                                    if ex.get("question"):
                                        st.markdown(f"*Question:* {ex['question']}")
                                    if ex.get("sql"):
                                        st.code(ex["sql"], language="sql")
                                else:
                                    st.text(str(ex))
                                if i < len(examples) - 1:
                                    st.divider()
                        except Exception:
                            st.text(str(item["question_sql_examples"]))
    else:
        st.info("No recent LLM queries found.")


def _render_user_activity_dialog_body(item: dict) -> None:
    """Render the body of the User Activity audit dialog (#157).

    Split out from the ``@st.dialog``-decorated wrapper so tests can call the
    body directly with a stubbed ``st`` module. The decorated wrapper
    ``_render_user_activity_dialog`` is what production calls.

    Surfaces ``user_agent`` for the first time (it's on the model but was
    dropped by the retired ``get_recent_activity`` helper). The
    "View user in Manage Users →" button is gated on a non-null ``user_id``
    because failed-login rows can have null ``user_id``.
    """
    # Header line: created_at · username · activity_type
    st.markdown(
        f"**{item.get('created_at')}** · {item.get('username') or '(unknown user)'} · {item.get('activity_type')}"
    )

    # Description
    description = item.get("description")
    st.write(description if description else "_(no description)_")

    # Request context — surfaces user_agent for the first time
    st.markdown(f"**IP Address:** {item.get('ip_address') or '_(unknown)_'}")
    st.markdown(f"**User Agent:** {item.get('user_agent') or '_(unknown)_'}")

    # Old/new JSON diff for settings changes.
    # Skip the section entirely when both old_value and new_value are null.
    old_val = item.get("old_value")
    new_val = item.get("new_value")
    if old_val or new_val:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Old Value**")
            if old_val:
                try:
                    st.json(json.loads(old_val))
                except Exception:
                    st.text(old_val)
            else:
                st.caption("_(none)_")
        with col2:
            st.markdown("**New Value**")
            if new_val:
                try:
                    st.json(json.loads(new_val))
                except Exception:
                    st.text(new_val)
            else:
                st.caption("_(none)_")

    # Outbound deep-link to Manage Users — gated on non-null user_id.
    # Failed-login rows have null user_id and would link to nothing useful.
    if item.get("user_id") is not None:
        st.divider()
        if st.button(
            "View user in Manage Users →",
            key=f"user_activity_dialog_goto_users_{item.get('id')}",
            type="primary",
        ):
            st.session_state["manage_users_pref_user_id"] = item["user_id"]
            # JS tab-selection shim — mirrors views/admin_analytics.py:333-353 but
            # targets the parent-doc "Users" tab. Streamlit's st.tabs has no
            # server-side API to change the active tab, and st.switch_page is a
            # no-op on the same page, so we click the parent doc's "Users" tab
            # button after the rerun lands. The consumer side of the contract
            # (`manage_users_pref_user_id` -> pre-populate `mu_selected_label`)
            # was added in #155 to views/admin_users.py.
            st.components.v1.html(
                """
                <script>
                  (function(){
                    try {
                      var tabs = window.parent.document.querySelectorAll('button[role=tab]');
                      for (var i = 0; i < tabs.length; i++) {
                        if (tabs[i].textContent.trim() === 'Users') { tabs[i].click(); break; }
                      }
                    } catch (e) { console.error('users-tab nav failed', e); }
                  })();
                </script>
                """,
                height=0,
            )
            st.switch_page("views/admin.py")

    # Metadata caption
    st.caption(f"Activity ID {item.get('id')}")


@st.dialog("User Activity Detail")
def _render_user_activity_dialog(item: dict) -> None:
    """``@st.dialog``-decorated wrapper invoked from ``_render_activity_tab``.

    Epic #169 / Feature #170 swapped the trigger primitive to
    ``st.data_editor`` + a labeled ``CheckboxColumn`` that auto-opens
    this dialog when exactly one row is ticked. See Epic #169
    Architecture Considerations.
    """
    _render_user_activity_dialog_body(item)


def _render_activity_tab(days_int: int):
    """Render the User Activity tab."""
    from orm.logging_functions import (
        get_activity_by_type,
        get_activity_over_time,
        get_activity_stats,
        get_user_activity_page,
    )

    stats = get_activity_stats(days=days_int)

    # KPI Row
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        _kpi_card("Logins Today", stats["logins_today"])
    with k2:
        _kpi_card("Failed Logins", stats["failed_logins"], f"Last {days_int} days")
    with k3:
        _kpi_card("Settings Changes", stats["settings_changes"])
    with k4:
        _kpi_card("Unique Users", stats["unique_users"], "Who logged in")

    st.divider()

    # Login Trends
    st.subheader("Login Trends")
    over_time = get_activity_over_time(days=days_int)
    if over_time:
        ot_df = pd.DataFrame(over_time)
        ot_df["date"] = pd.to_datetime(ot_df["date"])
        fig = px.line(
            ot_df,
            x="date",
            y=["logins", "failed_logins"],
            markers=True,
            title="Logins vs Failed Logins Over Time",
        )
        fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), legend_title_text="")
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No login activity data available yet.")

    st.divider()

    # Activity Type Breakdown (table only — Epic #144 cut the adjacent pie chart).
    st.subheader("Activity Type Breakdown")
    by_type = get_activity_by_type(days=days_int)
    if by_type:
        type_df = pd.DataFrame(by_type)
        st.dataframe(
            type_df.rename(columns={"activity_type": "Activity Type", "count": "Count"}),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("No activity type data available yet.")

    st.divider()

    # Recent Activity Log — paginated grid with single-row selection trigger
    # (mirrors the Questions tab primitive locked in by #155). No role gate
    # and no agent_logging.mode interaction: UserActivity rows don't carry
    # PHI or generated SQL, so the existing _guard_admin() is sufficient.
    st.subheader("Recent Activity Log")

    # Pagination controls (mirror the Questions tab shape)
    if "audit_activity_page_bump" in st.session_state:
        st.session_state["audit_activity_page_num"] = max(
            1,
            int(st.session_state.get("audit_activity_page_num", 1)) + int(st.session_state["audit_activity_page_bump"]),
        )
        del st.session_state["audit_activity_page_bump"]
    if "audit_activity_page_num" not in st.session_state:
        st.session_state["audit_activity_page_num"] = 1

    pc1, pc2, _spacer = st.columns([1, 1, 6])
    with pc1:
        page_size = st.selectbox(
            "Page size",
            options=[25, 50, 100, 200],
            index=1,
            key="audit_activity_page_size",
        )
    with pc2:
        st.number_input("Page", min_value=1, step=1, key="audit_activity_page_num")
        page = int(st.session_state["audit_activity_page_num"])

    result = get_user_activity_page(days=days_int, page=page, page_size=int(page_size))
    items = result.get("items", [])
    total = int(result.get("total", 0))
    total_pages = max(1, (total + int(page_size) - 1) // int(page_size))

    if items:
        # Epic #169 / Feature #170: same data_editor + labeled
        # CheckboxColumn auto-open pattern as the Questions tab. See the
        # ``_render_audit_trail_tab`` comments for the full rationale.
        table_rows = [
            {
                _VIEW_COLUMN_LABEL: False,
                "Timestamp": it["created_at"],
                "User": it.get("username") or "(unknown)",
                "Type": it.get("activity_type"),
                "Description": it.get("description"),
                "IP Address": it.get("ip_address"),
            }
            for it in items
        ]
        edited_df = st.data_editor(
            pd.DataFrame(table_rows),
            width="stretch",
            hide_index=True,
            num_rows="fixed",
            key="audit_activity_dataframe",
            column_config={
                _VIEW_COLUMN_LABEL: st.column_config.CheckboxColumn(
                    _VIEW_COLUMN_LABEL,
                    help=_VIEW_COLUMN_HELP,
                    default=False,
                    width="small",
                ),
                "Timestamp": st.column_config.TextColumn("Timestamp", disabled=True),
                "User": st.column_config.TextColumn("User", disabled=True),
                "Type": st.column_config.TextColumn("Type", disabled=True),
                "Description": st.column_config.TextColumn("Description", disabled=True),
                "IP Address": st.column_config.TextColumn("IP Address", disabled=True),
            },
        )

        try:
            current_checked = [int(i) for i in edited_df.index[edited_df[_VIEW_COLUMN_LABEL]].tolist()]
        except Exception:
            current_checked = []

        # Single-select enforcement (see Questions tab block for the full
        # rationale). Mutate ``data_editor``'s widget state to untick the
        # older row(s) and ``st.rerun()`` so the next render reflects the
        # single tick.
        prev_checked = list(st.session_state.get("audit_activity_prev_view_checks", []))
        if len(current_checked) > 1:
            newly_checked = [i for i in current_checked if i not in prev_checked]
            keep_idx = newly_checked[0] if newly_checked else current_checked[0]
            editor_state = st.session_state.get("audit_activity_dataframe")
            if isinstance(editor_state, dict):
                edited_rows = editor_state.setdefault("edited_rows", {})
                for idx in current_checked:
                    if idx == keep_idx:
                        continue
                    row_edits = edited_rows.get(idx)
                    if isinstance(row_edits, dict) and _VIEW_COLUMN_LABEL in row_edits:
                        del row_edits[_VIEW_COLUMN_LABEL]
                        if not row_edits:
                            del edited_rows[idx]
            st.session_state["audit_activity_prev_view_checks"] = [keep_idx]
            st.rerun()
        else:
            st.session_state["audit_activity_prev_view_checks"] = current_checked

        checked_count = len(current_checked)
        if checked_count == 1:
            row_idx = current_checked[0]
            if 0 <= row_idx < len(items):
                selected_item = items[row_idx]
                # Auto-open: dialog fires when exactly one row is ticked.
                # Gate on the tab-specific ``open_id`` session key so the
                # dialog doesn't re-fire on every rerun while the
                # checkbox stays ticked. Cross-tab claim guard from PR
                # #168 still applies as defense-in-depth.
                open_id = st.session_state.get("audit_activity_dialog_open_id")
                if open_id != selected_item["id"]:
                    if not st.session_state.get("_audit_dialog_claimed_this_rerun"):
                        st.session_state["_audit_dialog_claimed_this_rerun"] = True
                        st.session_state["audit_activity_dialog_open_id"] = selected_item["id"]
                        _render_user_activity_dialog(selected_item)
        elif checked_count == 0:
            # Allow re-ticking the same row to reopen the dialog by
            # clearing the gate.
            st.session_state.pop("audit_activity_dialog_open_id", None)
    else:
        st.info("No recent activity found.")

    # Pagination caption + Prev/Next
    cprev, cinfo, cnext = st.columns([1, 3, 1])
    with cprev:
        st.button(
            "Prev",
            disabled=int(page) <= 1,
            key="audit_activity_prev_btn",
            on_click=lambda: st.session_state.update({"audit_activity_page_bump": -1}),
        )
    with cinfo:
        st.caption(f"Page {int(page)} of {total_pages} • {total} total")
    with cnext:
        st.button(
            "Next",
            disabled=int(page) >= total_pages,
            key="audit_activity_next_btn",
            on_click=lambda: st.session_state.update({"audit_activity_page_bump": 1}),
        )


def _format_action_target(item: dict) -> str:
    """Format the target column for the Admin Actions grid.

    Prefers ``target_username`` (the human-readable handle, denormalised on
    the row so deleted users still render). Falls back to
    ``target_entity_type:target_entity_id`` when no username is set. Returns
    empty string when neither is present.
    """
    username = item.get("target_username")
    if username:
        return str(username)
    entity_type = item.get("target_entity_type")
    entity_id = item.get("target_entity_id")
    if entity_type and entity_id:
        return f"{entity_type}:{entity_id}"
    if entity_type:
        return str(entity_type)
    if entity_id:
        return str(entity_id)
    return ""


def _render_admin_action_dialog_body(item: dict) -> None:
    """Render the body of the Admin Action detail dialog (#156).

    Split out from the ``@st.dialog``-decorated wrapper so tests can call
    the body directly with a stubbed ``st`` module — mirrors the pattern
    used by ``_render_audit_question_dialog_body`` (#155).
    """
    # Header line: timestamp · admin · action_type
    created_at = item.get("created_at")
    if hasattr(created_at, "strftime"):
        created_str = created_at.strftime("%Y-%m-%d %H:%M:%S")
    else:
        created_str = str(created_at) if created_at is not None else "(unknown time)"
    admin = item.get("admin_username") or "Unknown"
    action_type = item.get("action_type") or "(unknown action)"
    st.markdown(f"**{created_str}** · {admin} · {action_type}")

    # Success / failure badge
    success = item.get("success")
    if success is False:
        st.markdown(":red[❌ FAILED]")
    else:
        st.markdown(":green[✅ SUCCESS]")

    # Description
    st.markdown("**Description**")
    st.write(item.get("description") or "_(no description)_")

    # Target line — surface fields the old expander dropped on the floor.
    target = _format_action_target(item)
    if target:
        st.markdown(f"**Target:** {target}")

    # Affected count (bulk operations) — only when non-null.
    affected = item.get("affected_count")
    if affected is not None:
        st.markdown(f"**Affected:** {int(affected)}")

    # Error block — only on failure with a non-empty message.
    error_msg = item.get("error_message")
    if success is False and error_msg:
        st.markdown("**Error**")
        st.code(str(error_msg), language="text")

    # Side-by-side old/new value diff.
    old_raw = item.get("old_value")
    new_raw = item.get("new_value")
    has_old = old_raw is not None and old_raw != ""
    has_new = new_raw is not None and new_raw != ""
    if has_old or has_new:
        col1, col2 = st.columns(2)
        if has_old:
            with col1:
                st.markdown("**Previous State**")
                try:
                    st.json(json.loads(old_raw))
                except Exception:
                    st.text(str(old_raw))
        if has_new:
            with col2:
                st.markdown("**New State**")
                try:
                    st.json(json.loads(new_raw))
                except Exception:
                    st.text(str(new_raw))

    # Trailing metadata caption.
    st.caption(f"Action ID {item.get('id')}")


@st.dialog("Admin Action Detail")
def _render_admin_action_dialog(item: dict) -> None:
    """``@st.dialog``-decorated wrapper invoked from ``_render_audit_tab``.

    Uses the same trigger primitive as the Questions tab: Epic #169 /
    Feature #170's ``st.data_editor`` + labeled ``CheckboxColumn``.
    Ticking exactly one row auto-opens this dialog (no button). See
    Epic #169.
    """
    _render_admin_action_dialog_body(item)


def _render_audit_tab(days_int: int):
    """Render the Admin Audit tab.

    Grid + row-click dialog (Feature #156). The pre-existing KPI row and
    Action Distribution pie/table at the top of the tab are unchanged from
    #144 — only the trailing "Recent Admin Actions" expander block was
    swapped for the paginated grid + dialog.
    """
    from orm.logging_functions import (
        get_admin_action_stats,
        get_admin_actions_by_type,
        get_admin_actions_page,
    )

    stats = get_admin_action_stats(days=days_int)

    # KPI Row
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        _kpi_card("Total Actions", stats["total"])
    with k2:
        _kpi_card("User Changes", stats["user_changes"], "Create/Update/Delete")
    with k3:
        _kpi_card("Training Actions", stats["training_actions"], "Approve/Reject")
    with k4:
        _kpi_card("Failed Actions", stats["failed"], "Unsuccessful")

    st.divider()

    # Action Distribution
    st.subheader("Action Distribution")
    by_type = get_admin_actions_by_type(days=days_int)
    if by_type:
        col1, col2 = st.columns(2)
        with col1:
            type_df = pd.DataFrame(by_type)
            fig = px.pie(type_df, values="count", names="action_type", title="Actions by Type")
            fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, width="stretch")
        with col2:
            st.dataframe(
                type_df.rename(columns={"action_type": "Action Type", "count": "Count"}),
                width="stretch",
                hide_index=True,
            )
    else:
        st.info("No admin action data available yet.")

    st.divider()

    # Recent Admin Actions — paginated grid + row-click dialog (#156).
    st.subheader("Recent Admin Actions")

    # Reset pagination on time-range change so the page index doesn't
    # outlive a shrunken result set.
    if st.session_state.get("audit_actions_days_signature") != int(days_int):
        st.session_state["audit_actions_days_signature"] = int(days_int)
        st.session_state["audit_actions_page_num"] = 1

    # Pagination controls (top row) — mirrors _render_audit_trail_tab.
    if "audit_actions_page_bump" in st.session_state:
        st.session_state["audit_actions_page_num"] = max(
            1,
            int(st.session_state.get("audit_actions_page_num", 1)) + int(st.session_state["audit_actions_page_bump"]),
        )
        del st.session_state["audit_actions_page_bump"]
    if "audit_actions_page_num" not in st.session_state:
        st.session_state["audit_actions_page_num"] = 1

    pc1, pc2, _spacer = st.columns([1, 1, 6])
    with pc1:
        page_size = st.selectbox(
            "Page size",
            options=[25, 50, 100, 200],
            index=1,
            key="audit_actions_page_size",
        )
    with pc2:
        st.number_input("Page", min_value=1, step=1, key="audit_actions_page_num")
        page = int(st.session_state["audit_actions_page_num"])

    result = get_admin_actions_page(days=int(days_int), page=page, page_size=int(page_size))
    items = result.get("items", [])
    total = int(result.get("total", 0))
    total_pages = max(1, (total + int(page_size) - 1) // int(page_size))

    if items:
        # Epic #169 / Feature #170: data_editor + labeled CheckboxColumn
        # auto-open on tick — see ``_render_audit_trail_tab`` for the
        # full rationale.
        table_rows = []
        for it in items:
            ts = it.get("created_at")
            ts_str = ts.strftime("%Y-%m-%d %H:%M:%S") if hasattr(ts, "strftime") else str(ts or "")
            table_rows.append(
                {
                    _VIEW_COLUMN_LABEL: False,
                    "Time": ts_str,
                    "Admin": it.get("admin_username") or "Unknown",
                    "Action Type": it.get("action_type") or "",
                    "Target": _format_action_target(it),
                    "Description": _truncate(it.get("description"), 120),
                    "Status": "✅ Success" if it.get("success") else "❌ Failed",
                }
            )

        edited_df = st.data_editor(
            pd.DataFrame(table_rows),
            width="stretch",
            hide_index=True,
            num_rows="fixed",
            key="audit_actions_dataframe",
            column_config={
                _VIEW_COLUMN_LABEL: st.column_config.CheckboxColumn(
                    _VIEW_COLUMN_LABEL,
                    help=_VIEW_COLUMN_HELP,
                    default=False,
                    width="small",
                ),
                "Time": st.column_config.TextColumn("Time", disabled=True),
                "Admin": st.column_config.TextColumn("Admin", disabled=True),
                "Action Type": st.column_config.TextColumn("Action Type", disabled=True),
                "Target": st.column_config.TextColumn("Target", disabled=True),
                "Description": st.column_config.TextColumn("Description", disabled=True),
                "Status": st.column_config.TextColumn("Status", disabled=True),
            },
        )

        try:
            current_checked = [int(i) for i in edited_df.index[edited_df[_VIEW_COLUMN_LABEL]].tolist()]
        except Exception:
            current_checked = []

        # Single-select enforcement (see Questions tab block for the full
        # rationale). Mutate ``data_editor``'s widget state to untick the
        # older row(s) and ``st.rerun()`` so the next render reflects the
        # single tick.
        prev_checked = list(st.session_state.get("audit_actions_prev_view_checks", []))
        if len(current_checked) > 1:
            newly_checked = [i for i in current_checked if i not in prev_checked]
            keep_idx = newly_checked[0] if newly_checked else current_checked[0]
            editor_state = st.session_state.get("audit_actions_dataframe")
            if isinstance(editor_state, dict):
                edited_rows = editor_state.setdefault("edited_rows", {})
                for idx in current_checked:
                    if idx == keep_idx:
                        continue
                    row_edits = edited_rows.get(idx)
                    if isinstance(row_edits, dict) and _VIEW_COLUMN_LABEL in row_edits:
                        del row_edits[_VIEW_COLUMN_LABEL]
                        if not row_edits:
                            del edited_rows[idx]
            st.session_state["audit_actions_prev_view_checks"] = [keep_idx]
            st.rerun()
        else:
            st.session_state["audit_actions_prev_view_checks"] = current_checked

        checked_count = len(current_checked)
        if checked_count == 1:
            row_idx = current_checked[0]
            if 0 <= row_idx < len(items):
                selected_item = items[row_idx]
                # Auto-open: dialog fires when exactly one row is ticked.
                # Gate on the tab-specific ``open_id`` session key so the
                # dialog doesn't re-fire on every rerun while the
                # checkbox stays ticked. Cross-tab claim guard from PR
                # #168 still applies as defense-in-depth.
                open_id = st.session_state.get("audit_actions_dialog_open_id")
                if open_id != selected_item["id"]:
                    if not st.session_state.get("_audit_dialog_claimed_this_rerun"):
                        st.session_state["_audit_dialog_claimed_this_rerun"] = True
                        st.session_state["audit_actions_dialog_open_id"] = selected_item["id"]
                        _render_admin_action_dialog(selected_item)
        elif checked_count == 0:
            # Allow re-ticking the same row to reopen the dialog by
            # clearing the gate.
            st.session_state.pop("audit_actions_dialog_open_id", None)
    else:
        st.info("No admin actions in the selected time range.")

    # Pagination caption + Prev/Next.
    cprev, cinfo, cnext = st.columns([1, 3, 1])
    with cprev:
        st.button(
            "Prev",
            disabled=int(page) <= 1,
            key="audit_actions_prev_btn",
            on_click=lambda: st.session_state.update({"audit_actions_page_bump": -1}),
        )
    with cinfo:
        st.caption(f"Page {int(page)} of {total_pages} • {total} total")
    with cnext:
        st.button(
            "Next",
            disabled=int(page) >= total_pages,
            key="audit_actions_next_btn",
            on_click=lambda: st.session_state.update({"audit_actions_page_bump": 1}),
        )


# ============== Main Entry Point ==============


def main():
    _guard_admin()

    st.title("Admin Analytics")
    st.caption("Usage, performance, and audit insights for administrators")

    # Global controls: time range and refresh button
    control_cols = st.columns([3, 1])
    with control_cols[0]:
        days = st.segmented_control(
            "Time Range",
            options=["7 days", "30 days", "90 days"],
            selection_mode="single",
            default="30 days",
        )
    with control_cols[1]:
        if st.button("Refresh Data", help="Clear cached data and reload metrics"):
            _read_metrics.clear()
            from views.errors import _load as _errors_load, _load_aggregates as _errors_load_aggregates

            _errors_load.clear()
            _errors_load_aggregates.clear()
            st.rerun()

    days_int = {"7 days": 7, "30 days": 30, "90 days": 90}.get(days or "30 days", 30)

    # Tab navigation. The Errors tab (#106) hosts what used to be the
    # standalone "Errors" page from the admin sidebar — moved here so admins
    # land on a single Analytics surface for usage, performance, audit, and
    # error visibility.
    from views.errors import render as render_errors_tab

    tabs = st.tabs(["Audit Trail", "Overview", "LLM Performance", "User Activity", "Errors", "Admin Audit"])

    with tabs[0]:
        _render_audit_trail_tab(days_int)

    with tabs[1]:
        _render_overview_tab(days_int)

    with tabs[2]:
        _render_llm_tab(days_int)

    with tabs[3]:
        _render_activity_tab(days_int)

    with tabs[4]:
        render_errors_tab(days_int)

    with tabs[5]:
        _render_audit_tab(days_int)


if __name__ == "__main__":
    main()
