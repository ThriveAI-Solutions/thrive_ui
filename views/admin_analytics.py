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
    from orm.logging_functions import get_activity_stats, get_error_stats, get_llm_stats

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
    error_stats = get_error_stats(days=days_int)

    # KPIs row - original + new
    k1, k2, k3, k4, k5, k6, k7, k8 = st.columns(8)
    df = _to_dense_days(over_time, days_int)
    with k1:
        _kpi_card("Users", users_count)
    with k2:
        _kpi_card("Questions", int(df["questions"].sum()))
    with k3:
        _kpi_card("Errors", int(df["errors"].sum()))
    with k4:
        _kpi_card("Active Users", active_users)
    with k5:
        _kpi_card("LLM Queries", llm_stats["total"], "SQL generations")
    with k6:
        _kpi_card("Avg Latency", f"{llm_stats['avg_latency_ms']:.0f}ms")
    with k7:
        _kpi_card("Logins Today", activity_stats["logins_today"])
    with k8:
        _kpi_card("Critical Errors", error_stats["critical"], "Requires attention")

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


def _render_audit_trail_tab(days_int: int):
    """Render the Question Audit Trail tab (#135)."""
    import datetime as _dt

    from orm.logging_functions import (
        get_question_audit_export,
        get_question_audit_page,
    )
    from orm.functions import get_all_users

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

    f1, f2, f3 = st.columns([0.30, 0.30, 0.40])
    with f1:
        usernames_sel = st.multiselect(
            "User", options=options["usernames"], key="audit_user_filter"
        )
    with f2:
        orgs_sel = st.multiselect(
            "Organization", options=options["orgs"], key="audit_org_filter"
        )
    with f3:
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

    # Table
    if items:
        table_rows = []
        for it in items:
            table_rows.append(
                {
                    "Asked At": it["asked_at"],
                    "User": it["username"],
                    "Organization": it.get("organization") or "(no org)",
                    "Question": _truncate(it["question"], 120),
                    "SQL": _truncate(it.get("sql_text"), 80),
                    "Status": _AUDIT_STATUS_EMOJI.get(it["status"], it["status"]),
                    "Elapsed (s)": round(float(it.get("elapsed_seconds") or 0.0), 3),
                }
            )
        st.dataframe(pd.DataFrame(table_rows), width="stretch", hide_index=True)
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

    # Row drilldown expanders
    for it in items:
        header = f"{it['asked_at']} • {it['username']} • {_truncate(it['question'], 80)}"
        with st.expander(header):
            st.markdown("**Question**")
            st.write(it["question"] or "_(empty)_")
            st.markdown("**Generated SQL**")
            st.code(it.get("sql_text") or "(no SQL)", language="sql")
            st.markdown("**Summary**")
            summary = it.get("summary_text")
            st.write(summary if summary else "_(no summary)_")
            df_preview = it.get("dataframe_preview")
            if df_preview:
                st.markdown("**DataFrame preview (first 5 rows)**")
                try:
                    df_obj = pd.read_json(df_preview).head(5)
                    st.dataframe(df_obj, width="stretch", hide_index=True)
                except Exception:
                    st.text(str(df_preview)[:1000])
            err = it.get("error_text")
            if err:
                st.markdown("**Error**")
                st.code(err, language="text")
            st.caption(
                f"Asked At {it['asked_at']} · Elapsed "
                f"{round(float(it.get('elapsed_seconds') or 0.0), 3)}s · Message ID {it.get('user_message_id')}"
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


def _render_activity_tab(days_int: int):
    """Render the User Activity tab."""
    from orm.logging_functions import (
        get_activity_by_type,
        get_activity_over_time,
        get_activity_stats,
        get_recent_activity,
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

    # Recent Activity Log
    st.subheader("Recent Activity Log")
    recent = get_recent_activity(days=days_int, limit=50)
    if recent:
        activity_df = pd.DataFrame(recent)
        # Format for display
        display_df = activity_df[["created_at", "username", "activity_type", "description", "ip_address"]].copy()
        display_df.columns = ["Timestamp", "User", "Type", "Description", "IP Address"]
        st.dataframe(display_df, width="stretch", hide_index=True)

        # Show details for settings changes
        settings_changes = [a for a in recent if a["old_value"] or a["new_value"]]
        if settings_changes:
            with st.expander("Settings Change Details", expanded=False):
                for change in settings_changes[:10]:
                    st.markdown(f"**{change['username']}** - {change['description']}")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Old Value:**")
                        try:
                            old = json.loads(change["old_value"]) if change["old_value"] else {}
                            st.json(old)
                        except Exception:
                            st.text(change["old_value"])
                    with col2:
                        st.markdown("**New Value:**")
                        try:
                            new = json.loads(change["new_value"]) if change["new_value"] else {}
                            st.json(new)
                        except Exception:
                            st.text(change["new_value"])
                    st.divider()
    else:
        st.info("No recent activity found.")


def _render_audit_tab(days_int: int):
    """Render the Admin Audit tab."""
    from orm.logging_functions import (
        get_admin_action_stats,
        get_admin_actions_by_type,
        get_recent_admin_actions,
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

    # Recent Actions Log
    st.subheader("Recent Admin Actions")
    recent = get_recent_admin_actions(days=days_int, limit=50)
    if recent:
        for action in recent:
            status_icon = ":green[SUCCESS]" if action["success"] else ":red[FAILED]"
            with st.expander(
                f"{status_icon} **{action['action_type']}** by {action['admin_username']} - {action['created_at'].strftime('%Y-%m-%d %H:%M')}",
                expanded=False,
            ):
                st.markdown(f"**Description:** {action['description']}")
                if action["target_username"]:
                    st.markdown(f"**Target User:** {action['target_username']}")

                # Show value changes if present
                if action["old_value"] or action["new_value"]:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Previous State:**")
                        try:
                            old = json.loads(action["old_value"]) if action["old_value"] else {}
                            st.json(old)
                        except Exception:
                            st.text(action["old_value"] or "N/A")
                    with col2:
                        st.markdown("**New State:**")
                        try:
                            new = json.loads(action["new_value"]) if action["new_value"] else {}
                            st.json(new)
                        except Exception:
                            st.text(action["new_value"] or "N/A")
    else:
        st.info("No recent admin actions found.")


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
