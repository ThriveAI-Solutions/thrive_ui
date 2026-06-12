"""Admin → Audit → Queries sub-tab (Epic #190, Phase 2).

The existing Questions tab (``views/admin_analytics._render_audit_trail_tab``)
shows one row per user question — fine for the legacy Vanna pipeline but
hides the multi-tool, multi-SQL reality of the agentic flow. This tab
renders one row per *query unit* sourced from
``orm.logging_functions.get_per_query_audit_page`` (Phase 1):

  * Legacy question → 1 row (the assistant SQL Message).
  * Agentic question → 1 row per ``ToolCall`` for the run, with the
    decoded ``sql_executed_json`` list rendered inside the row.

Two view modes back the same per-query row component:

  * **Flat** — single ``st.data_editor``; row-tick opens a per-query
    detail dialog; CSV export available.
  * **Grouped** — collapses rows by question into ``st.expander`` blocks,
    each showing a header (user, scope, total elapsed, query count) and
    a read-only table of the question's query units.

The Questions tab is left untouched here; Epic #190 Phase 4 retires (or
aliases) it.
"""

from __future__ import annotations

import datetime as _dt
import json
from typing import Any

import pandas as pd
import streamlit as st

from orm.logging_functions import (
    ALL_PIPELINES,
    ALL_SCOPES,
    SCOPE_LEGACY,
    get_per_query_audit_export,
    get_per_query_audit_page,
)
from utils.quick_logger import get_logger
from views.admin_analytics import (
    _AUDIT_STATUS_EMOJI,
    _VIEW_COLUMN_HELP,
    _VIEW_COLUMN_LABEL,
    _cached_audit_filter_options,
)

logger = get_logger(__name__)

# Cap each SQL preview to ~80 chars so the table column stays readable.
_SQL_PREVIEW_CHARS = 80

# Sentinels surfaced to the SQL column in the Flat-mode table. ``_DISABLED``
# matches the value the data layer already emits for ``logging_mode == 'disabled'``
# rows; we leave that alone. The scrubbed prefix is decorated here at render
# time — the data layer passes hashed content through verbatim.
_DISABLED_SQL_SENTINEL = "(logging disabled)"
_SCRUBBED_SQL_PREFIX = "(scrubbed) "
_NO_SQL_TOOL_RESULT = "(no SQL — tool result)"
_LEGACY_TOOL_LABEL = "(legacy SQL)"

_SCRUBBED_BANNER = (
    "Scrubbed mode — literals hashed. Verify against a full-fidelity environment before drawing conclusions."
)
_DISABLED_BANNER = "Logging disabled for this run — payload columns are not available."


# ---------------------------------------------------------------------------
# Row component (shared by Flat + Grouped + CSV export)
# ---------------------------------------------------------------------------


def _format_sql_preview(item: dict) -> str:
    """Return the SQL-column display string for a per-query item.

    Rules:
      * disabled mode → ``"(logging disabled)"``
      * agentic non-SQL tool call → ``"(no SQL — tool result)"``
      * scrubbed mode SQL → ``"(scrubbed) {first ~80 chars}"``
      * agentic SQL with multiple statements → ``"{n} statements: {first ~80}"``
      * legacy or single agentic SQL → first ~80 chars of the SQL text
      * no SQL at all → ``""``
    """
    mode = item.get("logging_mode")
    sql_statements = item.get("sql_statements") or []
    non_sql_summary = item.get("non_sql_summary")

    if mode == "disabled":
        # The data layer already sets non_sql_summary to the disabled sentinel.
        return _DISABLED_SQL_SENTINEL
    if item.get("pipeline") == "agentic" and not sql_statements:
        # Non-SQL tool call. The data layer routes the tool's result_summary
        # into ``non_sql_summary`` (e.g. KB hit metadata). Surface a stable
        # sentinel here regardless of the summary's shape so admins can spot
        # them at a glance.
        return _NO_SQL_TOOL_RESULT if non_sql_summary is not None else ""
    if not sql_statements:
        return ""
    first = (sql_statements[0] or "").strip().replace("\n", " ")
    truncated = first[:_SQL_PREVIEW_CHARS]
    if len(first) > _SQL_PREVIEW_CHARS:
        truncated += "…"
    if mode == "scrubbed":
        return f"{_SCRUBBED_SQL_PREFIX}{truncated}"
    if len(sql_statements) > 1:
        return f"{len(sql_statements)} statements: {truncated}"
    return truncated


def _format_patients_touched(item: dict) -> str:
    """Comma-separated list of ``source_id`` from ``patients_touched``."""
    touched = item.get("patients_touched") or []
    if not touched:
        return ""
    return ", ".join(p.get("source_id", "") for p in touched if p.get("source_id"))


def _derive_status(item: dict) -> str:
    """Return one of ``_AUDIT_STATUS_EMOJI`` strings.

    Rules mirror the Questions tab's status logic but adapted for per-query
    rows:
      * any ``error`` text → ❌ Error
      * any SQL or non_sql_summary or success=True → ✅ Success
      * otherwise → ⚪ Empty
    """
    if item.get("error"):
        return _AUDIT_STATUS_EMOJI["Error"]
    has_payload = bool(item.get("sql_statements")) or bool(item.get("non_sql_summary"))
    if item.get("success") is True or has_payload:
        return _AUDIT_STATUS_EMOJI["Success"]
    return _AUDIT_STATUS_EMOJI["Empty"]


def _per_query_row_to_table_dict(item: dict) -> dict[str, Any]:
    """Convert a Phase 1 per-query row dict into the table-display dict.

    Used by **both** Flat and Grouped modes so the column shape stays in
    lockstep. Flat mode keeps the ``View`` checkbox column; Grouped mode
    strips it before rendering.
    """
    return {
        _VIEW_COLUMN_LABEL: False,
        "Asked At": item["asked_at"],
        "User": item["username"],
        "Organization": item.get("organization") or "(no org)",
        "Pipeline": item.get("pipeline") or "",
        "Scope": item.get("scope") or SCOPE_LEGACY,
        "Tool": item.get("tool_name") or _LEGACY_TOOL_LABEL,
        "SQL": _format_sql_preview(item),
        "Patient(s)": _format_patients_touched(item),
        "Status": _derive_status(item),
        "Elapsed (ms)": int(item.get("elapsed_ms") or 0),
    }


def _group_items_by_question(items: list[dict]) -> list[tuple[dict, list[dict]]]:
    """Group per-query items by ``user_message_id`` preserving order.

    Returns a list of ``(group_header, units)`` tuples where ``group_header``
    summarises the question and ``units`` is the ordered list of per-query
    items belonging to it.
    """
    groups: list[tuple[dict, list[dict]]] = []
    by_msg: dict[int, int] = {}
    for it in items:
        mid = it.get("user_message_id")
        idx = by_msg.get(mid)
        if idx is None:
            header = {
                "user_message_id": mid,
                "asked_at": it.get("asked_at"),
                "username": it.get("username"),
                "organization": it.get("organization") or "(no org)",
                "question": it.get("question") or "",
                "scope": it.get("scope") or SCOPE_LEGACY,
                "total_elapsed_ms": int(it.get("elapsed_ms") or 0),
                "query_count": 1,
                "logging_mode": it.get("logging_mode"),
            }
            by_msg[mid] = len(groups)
            groups.append((header, [it]))
        else:
            header, units = groups[idx]
            header["total_elapsed_ms"] = int(header["total_elapsed_ms"]) + int(it.get("elapsed_ms") or 0)
            header["query_count"] = int(header["query_count"]) + 1
            # If any unit has a non-default logging mode, the group inherits
            # the most "alarming" one (disabled > scrubbed > full).
            modes_priority = {"disabled": 2, "scrubbed": 1, "full": 0, None: 0}
            cur_pri = modes_priority.get(header.get("logging_mode"), 0)
            new_pri = modes_priority.get(it.get("logging_mode"), 0)
            if new_pri > cur_pri:
                header["logging_mode"] = it.get("logging_mode")
            units.append(it)
    return groups


# ---------------------------------------------------------------------------
# Detail dialog bodies (separated for testability)
# ---------------------------------------------------------------------------


def _render_mode_banners(item: dict) -> None:
    """Render the scrubbed/disabled-mode warning banners for an item."""
    mode = item.get("logging_mode")
    if mode == "scrubbed":
        st.warning(_SCRUBBED_BANNER)
    elif mode == "disabled":
        st.warning(_DISABLED_BANNER)


def _render_sql_block(item: dict, *, can_see_query_details: bool) -> None:
    """Render the SQL statements section of the per-query dialog body."""
    sql_statements = item.get("sql_statements") or []
    non_sql_summary = item.get("non_sql_summary")
    mode = item.get("logging_mode")

    if not can_see_query_details:
        st.markdown("**SQL**")
        st.write("_(restricted — your role cannot see query details)_")
        return

    if mode == "disabled":
        st.markdown("**SQL**")
        st.write(f"_{_DISABLED_SQL_SENTINEL}_")
        return

    if sql_statements:
        if len(sql_statements) == 1:
            st.markdown("**SQL**")
            st.code(sql_statements[0], language="sql")
        else:
            for i, sql in enumerate(sql_statements, start=1):
                st.markdown(f"**SQL {i} of {len(sql_statements)}**")
                st.code(sql, language="sql")
        return

    if item.get("pipeline") == "agentic":
        st.markdown("**Tool result**")
        if non_sql_summary:
            st.code(non_sql_summary, language="json")
        else:
            st.write("_(no result captured)_")
        return

    # Legacy with no SQL → nothing to show.
    st.markdown("**SQL**")
    st.write("_(no SQL captured)_")


def _render_patients_block(item: dict) -> None:
    """Render the patients-touched chips."""
    touched = item.get("patients_touched") or []
    if not touched:
        return
    st.markdown("**Patient(s) touched**")
    cols = st.columns(min(4, len(touched)))
    for i, p in enumerate(touched):
        label = p.get("display_name") or p.get("source_id") or "(unknown)"
        sid = p.get("source_id") or ""
        with cols[i % len(cols)]:
            st.markdown(f"`{sid}` &nbsp; {label}")


def _render_query_detail_dialog_body(item: dict) -> None:
    """Body of the per-query detail dialog (Flat-mode click target)."""
    from agent.observability_gate import role_can_see_query_details

    current_role = st.session_state.get("user_role")
    can_see_query_details = role_can_see_query_details(current_role)

    # Header line
    org = item.get("organization") or "(no org)"
    st.markdown(f"**{item['asked_at']}** · {item['username']} · {org}")

    # Pipeline + Scope chips
    pipeline = item.get("pipeline") or "?"
    scope = item.get("scope") or SCOPE_LEGACY
    chips = [f"Pipeline: **{pipeline}**", f"Scope: **{scope}**"]
    if item.get("tool_name"):
        chips.append(f"Tool: **{item['tool_name']}**")
        if item.get("call_index") is not None:
            chips[-1] += f" (call #{item['call_index']})"
    st.markdown(" · ".join(chips))

    _render_mode_banners(item)

    st.markdown("**Question**")
    st.write(item.get("question") or "_(empty)_")

    _render_sql_block(item, can_see_query_details=can_see_query_details)

    _render_patients_block(item)

    err = item.get("error")
    if err:
        st.markdown("**Error**")
        st.code(err, language="text")

    # Metadata caption
    parts = []
    if item.get("run_id"):
        parts.append(f"run_id {item['run_id']}")
    if item.get("tool_call_id") is not None:
        parts.append(f"tool_call_id {item['tool_call_id']}")
    parts.append(f"message_id {item.get('user_message_id')}")
    parts.append(f"elapsed {int(item.get('elapsed_ms') or 0)}ms")
    st.caption(" · ".join(parts))

    # Outbound deep-link to Manage Users. Same contract as the retired
    # Questions tab's dialog (formerly in views/admin_analytics.py:482-507):
    # set ``manage_users_pref_user_id`` and switch_page so admin.py's
    # Manage Users surface pre-selects the row. JS shim clicks the outer
    # "Users" tab after the rerun lands. Key is per-(message, tool_call)
    # composite so two dialogs in the same rerun (shouldn't happen — cross
    # tab guard prevents it — but belt and braces) don't collide.
    st.divider()
    goto_users_key = f"queries_dialog_goto_users_{item.get('user_message_id')}_{item.get('tool_call_id')}"
    if st.button(
        "View user in Manage Users →",
        key=goto_users_key,
        type="primary",
    ):
        st.session_state["manage_users_pref_user_id"] = item["user_id"]
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


@st.dialog("Query Detail")
def _render_query_detail_dialog(item: dict) -> None:
    _render_query_detail_dialog_body(item)


def _render_question_detail_dialog_body(group_header: dict, units: list[dict]) -> None:
    """Body of the per-question detail dialog (Grouped-mode click target)."""
    st.markdown(
        f"**{group_header.get('asked_at')}** · {group_header.get('username')} · "
        f"{group_header.get('organization') or '(no org)'}"
    )
    st.markdown(
        f"Scope: **{group_header.get('scope')}** · "
        f"Query count: **{group_header.get('query_count')}** · "
        f"Total elapsed: **{int(group_header.get('total_elapsed_ms') or 0)}ms**"
    )
    if group_header.get("logging_mode") in {"scrubbed", "disabled"}:
        _render_mode_banners({"logging_mode": group_header["logging_mode"]})

    st.markdown("**Question**")
    st.write(group_header.get("question") or "_(empty)_")
    st.divider()

    for i, unit in enumerate(units):
        st.markdown(f"### Query {i + 1} of {len(units)}")
        _render_query_detail_dialog_body(unit)
        if i + 1 < len(units):
            st.divider()


@st.dialog("Question Detail")
def _render_question_detail_dialog(group_header: dict, units: list[dict]) -> None:
    _render_question_detail_dialog_body(group_header, units)


# ---------------------------------------------------------------------------
# Tab body
# ---------------------------------------------------------------------------


def _render_filters_row(days_int: int, options: dict) -> dict:
    """Render the filter row and return the assembled ``filters`` dict."""
    f1, f2, f3, f4, f5 = st.columns([0.2, 0.2, 0.2, 0.15, 0.25])
    with f1:
        usernames_sel = st.multiselect(
            "User",
            options=options["usernames"],
            key="queries_user_filter",
        )
    with f2:
        orgs_sel = st.multiselect(
            "Organization",
            options=options["orgs"],
            key="queries_org_filter",
        )
    with f3:
        scopes_sel = st.multiselect(
            "Scope",
            options=list(ALL_SCOPES),
            key="queries_scope_filter",
            help=(
                "Patient = single-patient questions (PHI access). "
                "Pop Health = cohort searches. Other = codes / KB / pure SQL. "
                "Legacy/Unknown = pre-agentic-replatform rows."
            ),
        )
    with f4:
        pipelines_sel = st.multiselect(
            "Pipeline",
            options=list(ALL_PIPELINES),
            key="queries_pipeline_filter",
            help="Filter by legacy Vanna pipeline rows vs new agentic pipeline rows.",
        )
    with f5:
        search = st.text_input(
            "Search question or SQL",
            placeholder="Substring match (case-insensitive)",
            key="queries_search",
        )

    return {
        "usernames": usernames_sel,
        "orgs": orgs_sel,
        "days": int(days_int),
        "search": search.strip() if search else None,
        "scopes": sorted(scopes_sel) if scopes_sel else None,
        "pipelines": sorted(pipelines_sel) if pipelines_sel else None,
    }


def _reset_pagination_on_filter_change(filters: dict, days_int: int, mode: str, key_prefix: str = "queries") -> None:
    sig_key = f"{key_prefix}_filter_signature"
    page_key = f"{key_prefix}_page_num"
    editor_key = f"{key_prefix}_dataframe"
    dialog_key = f"{key_prefix}_dialog_open_id"
    sig = json.dumps({**filters, "_days": days_int, "_mode": mode}, sort_keys=True, default=str)
    if st.session_state.get(sig_key) != sig:
        st.session_state[sig_key] = sig
        st.session_state[page_key] = 1
        # Clear stale editor state so an old tick can't fire the dialog
        # against a now-different item.
        editor_state = st.session_state.get(editor_key)
        if isinstance(editor_state, dict):
            editor_state["edited_rows"] = {}
        st.session_state.pop(dialog_key, None)


def _render_pagination_top(key_prefix: str = "queries") -> tuple[int, int]:
    bump_key = f"{key_prefix}_page_bump"
    page_key = f"{key_prefix}_page_num"
    size_key = f"{key_prefix}_page_size"
    if bump_key in st.session_state:
        st.session_state[page_key] = max(
            1,
            int(st.session_state.get(page_key, 1)) + int(st.session_state[bump_key]),
        )
        del st.session_state[bump_key]
    if page_key not in st.session_state:
        st.session_state[page_key] = 1

    pc1, pc2, _spacer = st.columns([1, 1, 6])
    with pc1:
        page_size = st.selectbox(
            "Page size",
            options=[25, 50, 100, 200],
            index=1,
            key=size_key,
        )
    with pc2:
        st.number_input("Page", min_value=1, step=1, key=page_key)
    return int(page_size), int(st.session_state[page_key])


def _render_pagination_bottom(page: int, total_pages: int, total: int, key_prefix: str = "queries") -> None:
    prev_key = f"{key_prefix}_prev_btn"
    next_key = f"{key_prefix}_next_btn"
    bump_key = f"{key_prefix}_page_bump"
    cprev, cinfo, cnext = st.columns([1, 3, 1])
    with cprev:
        st.button(
            "Prev",
            disabled=int(page) <= 1,
            key=prev_key,
            on_click=lambda: st.session_state.update({bump_key: -1}),
        )
    with cinfo:
        st.caption(f"Page {int(page)} of {total_pages} • {total} total")
    with cnext:
        st.button(
            "Next",
            disabled=int(page) >= total_pages,
            key=next_key,
            on_click=lambda: st.session_state.update({bump_key: 1}),
        )


def _render_flat_mode(items: list[dict], *, selection_enabled: bool, key_prefix: str = "queries") -> None:
    """Render Flat mode: single data_editor with row-tick → detail dialog."""
    if not items:
        st.info("No queries match the current filters.")
        return

    table_rows = [_per_query_row_to_table_dict(it) for it in items]

    editor_key = f"{key_prefix}_dataframe"
    prev_checks_key = f"{key_prefix}_prev_view_checks"
    dialog_key = f"{key_prefix}_dialog_open_id"

    if not selection_enabled:
        # agent_logging.mode = "disabled" → still surface rows (the data
        # layer sentinels payloads) but with no click affordance.
        readonly_rows = [{k: v for k, v in row.items() if k != _VIEW_COLUMN_LABEL} for row in table_rows]
        st.dataframe(
            pd.DataFrame(readonly_rows),
            width="stretch",
            hide_index=True,
            key=editor_key,
        )
        return

    edited_df = st.data_editor(
        pd.DataFrame(table_rows),
        width="stretch",
        hide_index=True,
        num_rows="fixed",
        key=editor_key,
        column_config={
            _VIEW_COLUMN_LABEL: st.column_config.CheckboxColumn(
                _VIEW_COLUMN_LABEL,
                help=_VIEW_COLUMN_HELP,
                default=False,
                width="small",
            ),
            # ``DatetimeColumn`` so the timestamp renders as a localised
            # "YYYY-MM-DD HH:MM:SS" instead of the raw int-ms-since-epoch
            # that ``TextColumn`` falls back to when given a datetime.
            "Asked At": st.column_config.DatetimeColumn("Asked At", format="YYYY-MM-DD HH:mm:ss", disabled=True),
            "User": st.column_config.TextColumn("User", disabled=True),
            "Organization": st.column_config.TextColumn("Organization", disabled=True),
            "Pipeline": st.column_config.TextColumn("Pipeline", disabled=True),
            "Scope": st.column_config.TextColumn("Scope", disabled=True),
            "Tool": st.column_config.TextColumn("Tool", disabled=True),
            "SQL": st.column_config.TextColumn("SQL", disabled=True),
            "Patient(s)": st.column_config.TextColumn("Patient(s)", disabled=True),
            "Status": st.column_config.TextColumn("Status", disabled=True),
            "Elapsed (ms)": st.column_config.NumberColumn("Elapsed (ms)", disabled=True),
        },
    )

    try:
        current_checked = [int(i) for i in edited_df.index[edited_df[_VIEW_COLUMN_LABEL]].tolist()]
    except Exception:
        current_checked = []

    # Single-select enforcement (mirrors the Questions tab's pattern,
    # views/admin_analytics.py:684-702).
    prev_checked = list(st.session_state.get(prev_checks_key, []))
    if len(current_checked) > 1:
        newly_checked = [i for i in current_checked if i not in prev_checked]
        keep_idx = newly_checked[0] if newly_checked else current_checked[0]
        editor_state = st.session_state.get(editor_key)
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
        st.session_state[prev_checks_key] = [keep_idx]
        st.rerun()
    else:
        st.session_state[prev_checks_key] = current_checked

    checked_count = len(current_checked)
    if checked_count == 1:
        row_idx = current_checked[0]
        if 0 <= row_idx < len(items):
            selected_item = items[row_idx]
            row_key = _flat_dialog_open_key(selected_item)
            open_id = st.session_state.get(dialog_key)
            if open_id != row_key:
                if not st.session_state.get("_audit_dialog_claimed_this_rerun"):
                    st.session_state["_audit_dialog_claimed_this_rerun"] = True
                    st.session_state[dialog_key] = row_key
                    _render_query_detail_dialog(selected_item)
    elif checked_count == 0:
        st.session_state.pop(dialog_key, None)


def _flat_dialog_open_key(item: dict) -> str:
    """Composite key identifying which per-query row is currently open in the
    Flat-mode detail dialog. Distinct per (message, tool_call) so two rows of
    the same question can each open the dialog independently."""
    return f"{item.get('user_message_id')}:{item.get('tool_call_id')}"


def _render_grouped_mode(items: list[dict], key_prefix: str = "queries") -> None:
    """Render Grouped mode: one ``st.expander`` per question."""
    if not items:
        st.info("No queries match the current filters.")
        return

    dialog_key = f"{key_prefix}_dialog_open_id"
    groups = _group_items_by_question(items)
    for header, units in groups:
        title = (
            f"{header['asked_at']} · {header['username']} · "
            f"{header['scope']} · {header['query_count']} querie(s) · "
            f"{int(header['total_elapsed_ms'])}ms"
        )
        with st.expander(title, expanded=False):
            if header.get("logging_mode") in {"scrubbed", "disabled"}:
                _render_mode_banners({"logging_mode": header["logging_mode"]})
            st.markdown(f"**Question:** {header.get('question') or '_(empty)_'}")

            unit_rows = [
                {k: v for k, v in _per_query_row_to_table_dict(u).items() if k != _VIEW_COLUMN_LABEL} for u in units
            ]
            st.dataframe(pd.DataFrame(unit_rows), width="stretch", hide_index=True)

            btn_key = f"{key_prefix}_group_detail_btn_{header['user_message_id']}"
            if st.button("View question detail →", key=btn_key):
                detail_key = f"q:{header['user_message_id']}"
                if (
                    not st.session_state.get("_audit_dialog_claimed_this_rerun")
                    and st.session_state.get(dialog_key) != detail_key
                ):
                    st.session_state["_audit_dialog_claimed_this_rerun"] = True
                    st.session_state[dialog_key] = detail_key
                    _render_question_detail_dialog(header, units)


def _render_csv_export(filters: dict, key_prefix: str = "queries") -> None:
    """Render the Flat-mode CSV export affordance."""
    btn_key = f"{key_prefix}_export_btn"
    download_key = f"{key_prefix}_export_download"
    if not st.button("📥 Export filtered queries to CSV", key=btn_key):
        return
    try:
        export_rows = get_per_query_audit_export(filters)
    except Exception as e:
        st.error(f"Export failed: {e}")
        logger.error("Per-query audit export failed: %s", e)
        return

    if not export_rows:
        st.warning("No queries to export with current filters.")
        return

    export_df = pd.DataFrame(
        [
            {
                "Asked At": r["asked_at"],
                "User": r["username"],
                "Organization": r.get("organization") or "(no org)",
                "Question": r["question"],
                "Pipeline": r.get("pipeline") or "",
                "Scope": r.get("scope") or SCOPE_LEGACY,
                "Tool": r.get("tool_name") or _LEGACY_TOOL_LABEL,
                # Full SQL list joined with a separator so the CSV is a single
                # cell per query unit. Admins who want a per-statement view can
                # open the Flat-mode dialog.
                "SQL": " ;; ".join(r.get("sql_statements") or []),
                "Non-SQL summary": r.get("non_sql_summary") or "",
                "Patient(s)": _format_patients_touched(r),
                "Logging mode": r.get("logging_mode") or "",
                "Status": _derive_status(r),
                "Elapsed (ms)": int(r.get("elapsed_ms") or 0),
            }
            for r in export_rows
        ]
    )
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    st.download_button(
        "Download per_query_audit_*.csv",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name=f"per_query_audit_{ts}.csv",
        mime="text/csv",
        key=download_key,
    )


def _render_queries_tab(days_int: int) -> None:
    try:
        logging_mode = st.secrets.get("agent_logging", {}).get("mode", "full")
    except Exception:
        logging_mode = "full"
    selection_enabled = logging_mode != "disabled"

    options = _cached_audit_filter_options(days_int)

    # Cross-Epic deep-link prefill (inherited from the retired Questions
    # tab — see views/admin_analytics.py:545-553 for the original). Honour
    # both the legacy ``audit_trail_pref_user_id`` key (chat_bot.py,
    # admin_users.py set this) and the forward-compat
    # ``audit_queries_pref_user_id`` for surfaces that adopt the new name.
    pref_user_id = st.session_state.pop("audit_trail_pref_user_id", None)
    if pref_user_id is None:
        pref_user_id = st.session_state.pop("audit_queries_pref_user_id", None)
    if pref_user_id is not None and "queries_user_filter" not in st.session_state:
        try:
            from orm.functions import get_all_users

            all_users = get_all_users()
            match = next((u for u in all_users if u["id"] == int(pref_user_id)), None)
            if match and match["username"] in options["usernames"]:
                st.session_state["queries_user_filter"] = [match["username"]]
        except Exception as e:
            logger.warning("Queries tab deep-link prefill failed: %s", e)

    filters = _render_filters_row(days_int, options)

    mc1, _spacer = st.columns([1, 7])
    with mc1:
        mode = st.radio(
            "View",
            options=["Grouped", "Flat"],
            index=0,
            horizontal=True,
            key="queries_view_mode",
        )

    _reset_pagination_on_filter_change(filters, days_int, mode)

    page_size, page = _render_pagination_top()

    try:
        result = get_per_query_audit_page(filters, page=page, page_size=int(page_size))
    except Exception as e:
        st.error(f"Failed to load queries: {e}")
        logger.warning("get_per_query_audit_page failed in view: %s", e)
        return

    items = result.get("items", [])
    total = int(result.get("total", 0))
    total_pages = max(1, (total + int(page_size) - 1) // int(page_size))

    if mode == "Flat":
        _render_flat_mode(items, selection_enabled=selection_enabled)
    else:
        _render_grouped_mode(items)

    _render_pagination_bottom(page, total_pages, total)

    if mode == "Flat":
        _render_csv_export(filters)


def render(days_int: int) -> None:
    _render_queries_tab(days_int)
