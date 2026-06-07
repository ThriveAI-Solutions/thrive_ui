"""Admin Errors panel — unified view across thrive_error_log,
thrive_agent_run failures, and the JSONL fallback file.

Originally a standalone Streamlit page; now mounted as the "Errors" tab
inside :mod:`views.admin_analytics`. The public entry point is
:func:`render`, which takes the time-range integer from the parent
Admin Analytics segmented control (no longer renders its own radio).

Consumes :func:`orm.error_read_model.query_errors` and
:func:`orm.error_read_model.count_errors_by_source` plus
:func:`orm.error_read_model.query_aggregates` for the KPI / chart block,
along with the pure helpers in :mod:`views.errors_helpers`.
"""

from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from orm.error_read_model import ErrorSource, query_aggregates
from orm.models import ErrorCategory, ErrorSeverity
from utils.error_fallback_sink import try_drain_fallback_to_db
from utils.quick_logger import get_logger
from views.errors_helpers import (
    MAX_ROW_LIMIT,
    _export_filename,
    _format_results_table,
    _pretty_context_data,
    _query_filtered,
    _time_range_to_since,
)

# Known categories include every ErrorCategory enum value plus the synthetic
# "agent_run" used for thrive_agent_run failures (see orm.error_read_model).
_KNOWN_CATEGORIES = sorted([c.value for c in ErrorCategory] + ["agent_run"])
_KNOWN_SEVERITIES = [s.value for s in ErrorSeverity]

# Admins often watch errors arrive in near-real-time, so the TTL is
# short compared to admin_analytics's 300s.
ERRORS_CACHE_TTL_SECONDS = 60
# Aggregate KPIs + chart series tolerate a longer cache so toggling filters
# doesn't re-roll up every row. Mirrors admin_analytics.ANALYTICS_CACHE_TTL_SECONDS.
AGGREGATES_CACHE_TTL_SECONDS = 300

# Pie-slice colors match the legacy Errors by Severity chart from
# views/admin_analytics.py so admins don't relearn the visual encoding.
_SEVERITY_COLOR_MAP = {
    "warning": "#FFA500",
    "error": "#FF6347",
    "critical": "#DC143C",
}

logger = get_logger(__name__)


def _kpi_card(
    label: str,
    value,
    help_text: str | None = None,
    dim: bool = False,
) -> None:
    """Render one source-count badge.

    When ``dim=True`` (the source is excluded by the Sources chip filter
    above), the label is struck through and the value rendered at 50%
    opacity so the admin sees both the per-source total *and* that the
    source is currently filtered out of the results list.
    """
    c = st.container(border=True)
    with c:
        if dim:
            st.markdown(f"**~~{label}~~**")
            st.markdown(
                f"<h3 style='margin-top:0; opacity:0.5'>{value}</h3>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(f"**{label}**")
            st.markdown(f"<h3 style='margin-top:0'>{value}</h3>", unsafe_allow_html=True)
        if help_text:
            st.caption(help_text)


@st.cache_data(ttl=ERRORS_CACHE_TTL_SECONDS, show_spinner="Loading errors...")
def _load(
    days: int,
    sources_csv: str,
    categories_csv: str,
    severities_csv: str,
    search: str,
    user_id_text: str,
) -> tuple[list[dict], dict[str, int]]:
    """Thin ``@st.cache_data`` wrapper around :func:`_query_filtered`.

    All inputs are stringly typed so the resulting tuple stays hashable
    for Streamlit's cache key.
    """
    return _query_filtered(
        days,
        sources_csv,
        categories_csv,
        severities_csv,
        search,
        user_id_text,
    )


@st.cache_data(ttl=AGGREGATES_CACHE_TTL_SECONDS, show_spinner=False)
def _load_aggregates(days: int) -> dict:
    """Roll up KPI + chart data for the Analytics block.

    Cached at 300s — chart data churns less than the per-row drill-down.
    Returns a plain dict (not the dataclass) so Streamlit's cache key
    handling does not need to fingerprint a dataclass.
    """
    agg = query_aggregates(_time_range_to_since(days))
    return {
        "total": agg.total,
        "critical": agg.critical,
        "sql_errors": agg.sql_errors,
        "retry_success_rate": agg.retry_success_rate,
        "over_time_by_category": agg.over_time_by_category,
        "by_category": agg.by_category,
        "by_severity": agg.by_severity,
    }


# ── Render entry point (called from views.admin_analytics tab) ────────────


def render(days_int: int) -> None:
    """Render the Errors tab inside Admin Analytics.

    ``days_int`` is the integer time-range value supplied by Admin
    Analytics's global segmented control (7 / 30 / 90). The Errors tab
    no longer owns its own time-range picker — it inherits the parent's.
    """
    st.caption("Unified view across the application database, agentic-run failures, and the durable fallback file.")

    source_options = [s.value for s in ErrorSource]
    selected_source_values = st.multiselect(
        "Sources",
        options=source_options,
        default=source_options,
        key="errors_sources",
    )
    sources_csv = ",".join(sorted(selected_source_values))

    with st.expander("Filters", expanded=False):
        filter_cols = st.columns(2)
        with filter_cols[0]:
            selected_categories = st.multiselect(
                "Categories",
                options=_KNOWN_CATEGORIES,
                default=[],
                key="errors_categories",
                help="Leave empty to include all categories.",
            )
            search_text = st.text_input(
                "Search (error message or question)",
                value="",
                key="errors_search",
                help="Case-insensitive substring match. Literal % and _ are matched.",
            )
        with filter_cols[1]:
            selected_severities = st.multiselect(
                "Severities",
                options=_KNOWN_SEVERITIES,
                default=[],
                key="errors_severities",
                help="Leave empty to include all severities.",
            )
            user_id_text = st.text_input(
                "User ID",
                value="",
                key="errors_user_id",
                help="Optional. Leave empty to include all users.",
            )

    categories_csv = ",".join(sorted(selected_categories))
    severities_csv = ",".join(sorted(selected_severities))

    rows_dicts, counts = _load(
        days_int,
        sources_csv,
        categories_csv,
        severities_csv,
        search_text,
        user_id_text,
    )

    kpi_help_text = (
        "Total in the selected time range. Not affected by category / severity / "
        "user / search filters — those narrow the table below only."
    )

    cols = st.columns(3)
    with cols[0]:
        _kpi_card(
            "Error Log",
            counts.get(ErrorSource.ERROR_LOG.value, 0),
            help_text=kpi_help_text,
            dim=ErrorSource.ERROR_LOG.value not in selected_source_values,
        )
    with cols[1]:
        _kpi_card(
            "Agent Runs",
            counts.get(ErrorSource.AGENT_RUN.value, 0),
            help_text=kpi_help_text,
            dim=ErrorSource.AGENT_RUN.value not in selected_source_values,
        )
    with cols[2]:
        _kpi_card(
            "Fallback File",
            counts.get(ErrorSource.FALLBACK_SINK.value, 0),
            help_text=kpi_help_text,
            dim=ErrorSource.FALLBACK_SINK.value not in selected_source_values,
        )

    st.divider()

    # ── Analytics block (migrated from the original Error Analysis tab) ──

    aggregates = _load_aggregates(days_int)

    st.subheader("Analytics")
    st.caption(
        "Time-range totals across all three sources. Not narrowed by the "
        "category / severity / user / search filters above."
    )

    a1, a2, a3, a4 = st.columns(4)
    with a1:
        _kpi_card("Total Errors", aggregates["total"])
    with a2:
        _kpi_card("Critical", aggregates["critical"], help_text="Severity = critical")
    with a3:
        _kpi_card("SQL Errors", aggregates["sql_errors"], help_text="Generation + Execution")
    with a4:
        _kpi_card("Retry Success", f"{aggregates['retry_success_rate']}%")

    st.markdown("**Error Trends by Category**")
    over_time = aggregates["over_time_by_category"]
    if over_time:
        ot_df = pd.DataFrame(over_time)
        ot_df["date"] = pd.to_datetime(ot_df["date"])
        pivot_df = ot_df.pivot(index="date", columns="category", values="count").fillna(0).reset_index()
        trends_fig = px.area(pivot_df, x="date", y=pivot_df.columns[1:], title=None)
        trends_fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), legend_title_text="Category")
        st.plotly_chart(trends_fig, width="stretch")
    else:
        st.info("No error time-series data in this range.")

    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.markdown("**Errors by Category**")
        by_category = aggregates["by_category"]
        if by_category:
            cat_df = pd.DataFrame(
                sorted(by_category.items(), key=lambda kv: kv[1], reverse=True),
                columns=["Category", "Count"],
            )
            cat_fig = px.bar(cat_df, x="Count", y="Category", orientation="h")
            cat_fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(cat_fig, width="stretch")
        else:
            st.info("No category data.")

    with chart_cols[1]:
        st.markdown("**Errors by Severity**")
        by_severity = aggregates["by_severity"]
        if by_severity:
            sev_df = pd.DataFrame(by_severity)
            sev_fig = px.pie(
                sev_df,
                values="count",
                names="severity",
                color="severity",
                color_discrete_map=_SEVERITY_COLOR_MAP,
            )
            sev_fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(sev_fig, width="stretch")
        else:
            st.info("No severity data.")

    st.divider()

    caption_text = f"Showing {len(rows_dicts)} error(s) under the current filters."
    if len(rows_dicts) >= MAX_ROW_LIMIT:
        caption_text += f" (capped at {MAX_ROW_LIMIT} — narrow the time range or apply more filters to see older rows)"
    st.caption(caption_text)

    export_payload = json.dumps(rows_dicts, indent=2)
    export_name = _export_filename(_time_range_to_since(days_int), sources_csv)
    st.download_button(
        label="⬇️ Export filtered errors as JSON",
        data=export_payload,
        file_name=export_name,
        mime="application/json",
        disabled=not rows_dicts,
        help="Downloads the currently visible rows with every field — not the entire table.",
    )

    if not rows_dicts:
        st.info("No errors match the current filters in the selected range.")
    else:
        for row in rows_dicts:
            sev = (row.get("severity") or "?").upper()
            cat = row.get("category") or "?"
            et = row.get("error_type") or "?"
            when = (row.get("created_at") or "")[:19].replace("T", " ")
            msg = row.get("error_message") or ""
            msg_preview = (msg[:80] + "…") if len(msg) > 80 else msg
            label = f"{sev} · {cat} · {et} · {when}"
            if msg_preview:
                label = f"{label} — {msg_preview}"

            with st.expander(label):
                left, right = st.columns([2, 1])
                with left:
                    st.markdown("**Error message**")
                    st.code(row.get("error_message") or "(none)", language="text")

                    if row.get("question"):
                        st.markdown("**Question**")
                        st.write(row["question"])

                    if row.get("generated_sql"):
                        st.markdown("**Generated SQL**")
                        st.code(row["generated_sql"], language="sql")

                    if row.get("stack_trace"):
                        st.markdown("**Stack trace**")
                        st.code(row["stack_trace"], language="text")

                    ctx_pretty = _pretty_context_data(row.get("context_data"))
                    if ctx_pretty:
                        st.markdown("**Context data**")
                        st.code(ctx_pretty, language="json")

                with right:
                    st.markdown("**Metadata**")
                    st.write(f"Source: `{row.get('source', '?')}`")
                    st.write(f"Created: `{row.get('created_at', '?')}`")
                    if row.get("user_id") is not None:
                        st.write(f"User ID: `{row['user_id']}`")
                    if row.get("group_id"):
                        st.write(f"Group ID: `{row['group_id']}`")
                    if row.get("message_id") is not None:
                        st.write(f"Message ID: `{row['message_id']}`")
                    if row.get("llm_provider"):
                        st.write(f"LLM provider: `{row['llm_provider']}`")
                    if row.get("llm_model"):
                        st.write(f"LLM model: `{row['llm_model']}`")

                    if row.get("source") == "agent_run" and row.get("run_id"):
                        st.markdown("---")
                        st.markdown("**Agent run**")
                        st.write(f"Run ID: `{row['run_id']}`")
                        try:
                            st.page_link(
                                "views/agent_analytics.py",
                                label="Open Agentic Analytics →",
                                icon="🧠",
                            )
                        except Exception:
                            # st.page_link is only valid inside a registered
                            # Streamlit page — skip the link if the runtime
                            # isn't multipage-aware.
                            pass
                    elif row.get("run_id"):
                        st.write(f"Run ID: `{row['run_id']}`")

                    retry_fields = (
                        row.get("auto_retry_attempted"),
                        row.get("retry_successful"),
                        row.get("retry_count"),
                    )
                    if any(v is not None for v in retry_fields):
                        st.markdown("---")
                        st.markdown("**Retry**")
                        st.write(f"Attempted: `{row.get('auto_retry_attempted')}`")
                        st.write(f"Successful: `{row.get('retry_successful')}`")
                        st.write(f"Count: `{row.get('retry_count')}`")

    st.divider()
    with st.expander("Maintenance — drain pending fallback records"):
        fb_count = counts.get(ErrorSource.FALLBACK_SINK.value, 0)
        st.write(
            f"There are currently **{fb_count}** record(s) in the durable fallback "
            "file waiting to be replayed into `thrive_error_log`. These are records "
            "that fired while the app SQLite was unavailable."
        )
        if st.button(
            "Drain pending fallback records into the DB",
            disabled=fb_count == 0,
            type="secondary",
            key="errors_drain_button",
        ):
            try:
                n = try_drain_fallback_to_db()
                if n > 0:
                    st.success(f"Replayed {n} record(s) into thrive_error_log.")
                else:
                    st.info(
                        "No records were replayed — the fallback file was empty "
                        "or all records were duplicates of existing rows."
                    )
                # Bust the load caches so the per-source badges and the
                # analytics block both refresh.
                _load.clear()
                _load_aggregates.clear()
                st.rerun()
            except Exception as exc:  # pragma: no cover — defensive in UI
                logger.exception("Errors tab: try_drain_fallback_to_db raised")
                st.error(f"Drain failed: {exc}")

    # Helper reserved for a future migration replacing the inline expander rendering.
    _ = _format_results_table
