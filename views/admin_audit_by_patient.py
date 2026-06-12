"""Admin → Audit → By Patient sub-tab (Epic #190, Phase 3).

A focused pivot view: the admin picks one or more patients (by name or
``source_id``) and the tab renders every per-query audit row that touched
those patients. Only agentic rows can match — the legacy pipeline has no
structured ``AgentPatientAccess`` record, so legacy questions never appear
here. This is called out in the picker's help text.

Implementation note: rendering reuses the Phase 2 helpers from
``views.admin_audit_queries`` (threaded via the ``key_prefix`` argument)
so we don't fork the data_editor / expander / dialog code. Only the
patient picker is new in this phase.
"""

from __future__ import annotations

import streamlit as st

from orm.agent_logging_functions import get_patient_audit_autocomplete
from utils.quick_logger import get_logger
from views import admin_audit_queries as _q
from views.admin_analytics import ANALYTICS_CACHE_TTL_SECONDS

logger = get_logger(__name__)

_KEY_PREFIX = "by_patient"

# Cap on the autocomplete option list. Older patients can still be reached
# via the paste textarea — see _PASTE_HELP for the user-facing note.
_AUTOCOMPLETE_LIMIT = 500

_TAB_HELP = (
    "Search by patient name or `source_id` to see every query that touched "
    "that patient. Only agentic queries appear here — the legacy pipeline "
    "has no structured patient-touch record."
)
_PASTE_HELP = (
    "Paste one source ID per line for patients outside the autocomplete "
    "window (older than the date range, or beyond the 500-row cap)."
)


@st.cache_data(ttl=ANALYTICS_CACHE_TTL_SECONDS, show_spinner=False)
def _cached_patient_options(days_int: int) -> list[dict]:
    """Cached read of the autocomplete options. Cache key = ``days_int``."""
    return get_patient_audit_autocomplete(days=days_int, limit=_AUTOCOMPLETE_LIMIT)


def _format_option(option: dict) -> str:
    """Render a single autocomplete option label."""
    sid = option.get("source_id") or ""
    name = option.get("display_name")
    if name:
        return f"{name} ({sid})"
    return sid


def _parse_paste(text: str | None) -> list[str]:
    """Split a paste-textarea blob into a deduplicated, trimmed list of IDs."""
    if not text:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        sid = line.strip()
        if not sid or sid in seen:
            continue
        seen.add(sid)
        out.append(sid)
    return out


def _render_patient_picker(days_int: int) -> list[str]:
    """Render the patient picker and return the deduped list of source_ids."""
    options = _cached_patient_options(days_int)

    pc1, pc2 = st.columns([0.6, 0.4])
    with pc1:
        selected_options = st.multiselect(
            "Patients (recently touched)",
            options=options,
            format_func=_format_option,
            key=f"{_KEY_PREFIX}_picker",
            help=(
                "Type a name or source_id to filter the list. "
                f"Showing up to {_AUTOCOMPLETE_LIMIT} most-recently-touched "
                "patients in the selected time range."
            ),
        )
    with pc2:
        paste = st.text_area(
            "Or paste source IDs",
            key=f"{_KEY_PREFIX}_paste",
            help=_PASTE_HELP,
            height=80,
        )

    picked: list[str] = []
    seen: set[str] = set()
    for opt in selected_options or []:
        sid = opt.get("source_id") if isinstance(opt, dict) else None
        if sid and sid not in seen:
            seen.add(sid)
            picked.append(sid)
    for sid in _parse_paste(paste):
        if sid not in seen:
            seen.add(sid)
            picked.append(sid)

    if options:
        st.caption(
            f"Showing {len(options)} patient(s) from the last {days_int} day(s). "
            f"For older patients, paste their source_id."
        )
    return picked


def _render_by_patient_tab(days_int: int) -> None:
    st.caption(_TAB_HELP)

    try:
        logging_mode = st.secrets.get("agent_logging", {}).get("mode", "full")
    except Exception:
        logging_mode = "full"
    selection_enabled = logging_mode != "disabled"

    source_ids = _render_patient_picker(days_int)
    if not source_ids:
        st.info("Pick one or more patients above to see every query that touched them across the agentic pipeline.")
        return

    # Mode toggle (Grouped first because the admin usually wants the
    # question-level context when investigating a patient).
    mc1, _spacer = st.columns([1, 7])
    with mc1:
        mode = st.radio(
            "View",
            options=["Grouped", "Flat"],
            index=0,
            horizontal=True,
            key=f"{_KEY_PREFIX}_view_mode",
        )

    # Restrict to agentic rows — legacy rows can't match a structured
    # patient touch anyway (the data layer already excludes them when
    # source_ids is set), but pinning the pipeline filter here makes the
    # filter signature stable across page-loads.
    filters = {
        "usernames": [],
        "orgs": [],
        "days": int(days_int),
        "search": None,
        "scopes": None,
        "pipelines": ["agentic"],
        "source_ids": source_ids,
    }

    _q._reset_pagination_on_filter_change(filters, days_int, mode, key_prefix=_KEY_PREFIX)
    page_size, page = _q._render_pagination_top(key_prefix=_KEY_PREFIX)

    try:
        result = _q.get_per_query_audit_page(filters, page=page, page_size=int(page_size))
    except Exception as e:
        st.error(f"Failed to load patient queries: {e}")
        logger.warning("get_per_query_audit_page failed in by-patient view: %s", e)
        return

    items = result.get("items", [])
    total = int(result.get("total", 0))
    total_pages = max(1, (total + int(page_size) - 1) // int(page_size))

    if mode == "Flat":
        _q._render_flat_mode(items, selection_enabled=selection_enabled, key_prefix=_KEY_PREFIX)
    else:
        _q._render_grouped_mode(items, key_prefix=_KEY_PREFIX)

    _q._render_pagination_bottom(page, total_pages, total, key_prefix=_KEY_PREFIX)

    if mode == "Flat":
        _q._render_csv_export(filters, key_prefix=_KEY_PREFIX)


def render(days_int: int) -> None:
    _render_by_patient_tab(days_int)
