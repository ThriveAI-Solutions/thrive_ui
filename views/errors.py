"""Admin Errors page — unified view across thrive_error_log,
thrive_agent_run failures, and the JSONL fallback file.

Phase 1: skeleton + admin gate + time-range + source-count badges +
source filter chips + basic results table. Subsequent phases add the
category / severity / user / search filters (Phase 2), per-row drill-down
expanders (Phase 3), JSON export (Phase 4), and the fallback drain
button + smoke tests (Phase 5).

Consumes :func:`orm.error_read_model.query_errors` and
:func:`orm.error_read_model.count_errors_by_source` from PR #96, plus
the pure helpers in :mod:`views.errors_helpers`.
"""

from __future__ import annotations

import json

import streamlit as st

from orm.error_read_model import ErrorSource
from orm.models import ErrorCategory, ErrorSeverity, RoleTypeEnum
from utils.error_fallback_sink import try_drain_fallback_to_db
from views.errors_helpers import (
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


def _guard_admin() -> None:
    if st.session_state.get("user_role") != RoleTypeEnum.ADMIN.value:
        st.error("You don't have permission to view this page.")
        st.stop()


def _kpi_card(label: str, value) -> None:
    c = st.container(border=True)
    with c:
        st.markdown(f"**{label}**")
        st.markdown(f"<h3 style='margin-top:0'>{value}</h3>", unsafe_allow_html=True)


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


# ── Page render (runs every Streamlit script execution) ───────────────────


_guard_admin()

st.title("Errors")
st.caption("Unified view across the application database, agentic-run failures, and the durable fallback file.")

range_choice = st.radio(
    "Time Range",
    options=[7, 30, 90],
    index=1,  # default to 30 days
    format_func=lambda d: f"{d} days",
    horizontal=True,
    key="errors_days",
)

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
    range_choice,
    sources_csv,
    categories_csv,
    severities_csv,
    search_text,
    user_id_text,
)

cols = st.columns(3)
with cols[0]:
    _kpi_card("Error Log", counts.get(ErrorSource.ERROR_LOG.value, 0))
with cols[1]:
    _kpi_card("Agent Runs", counts.get(ErrorSource.AGENT_RUN.value, 0))
with cols[2]:
    _kpi_card("Fallback File", counts.get(ErrorSource.FALLBACK_SINK.value, 0))

st.divider()

st.caption(f"Showing {len(rows_dicts)} error(s) under the current filters.")

export_payload = json.dumps(rows_dicts, indent=2)
export_name = _export_filename(_time_range_to_since(range_choice), sources_csv)
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
            # Bust the load cache so the per-source badges refresh.
            _load.clear()
            st.rerun()
        except Exception as exc:  # pragma: no cover — defensive in UI
            st.error(f"Drain failed: {exc}")

# Helper reserved for a future migration replacing the inline expander rendering.
_ = _format_results_table
