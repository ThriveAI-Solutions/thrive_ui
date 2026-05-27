"""Agentic Analytics — full-fidelity visibility into agent runs.

Admin-only. Surfaces run-level rollups, per-run inspector (the QA payoff),
tool breakdowns, and patient-access tracking from the agentic logging tables.
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from orm.models import RoleTypeEnum
from orm.agent_logging_functions import (
    get_agent_run_detail,
    get_agent_run_stats,
    get_patient_access,
    get_recent_agent_runs,
    get_runs_over_time,
    get_tool_breakdown,
    log_agent_run_viewed,
    set_run_review_status,
)
from agent.logging_config import AgentLoggingConfig


def _guard_admin() -> None:
    if st.session_state.get("user_role") != RoleTypeEnum.ADMIN.value:
        st.error("Admin access required.")
        st.stop()


def _kpi(label: str, value, help_text: str | None = None) -> None:
    st.metric(label, value, help=help_text)


def _timeline_rows(detail: dict) -> list[dict]:
    """Return the run's events sorted by seq for the inspector."""
    return sorted(detail.get("events", []), key=lambda e: e["seq"])


def _render_runs_tab(days: int) -> None:
    stats = get_agent_run_stats(days)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        _kpi("Runs", stats["total_runs"])
    with c2:
        _kpi("Success rate", f"{stats['success_rate']}%")
    with c3:
        _kpi("Cap-reached", f"{stats['cap_rate']}%")
    with c4:
        _kpi("Avg tools/run", stats["avg_tool_calls"])
    with c5:
        _kpi("Total tokens", f"{stats['total_tokens']:,}")

    over_time = get_runs_over_time(days)
    if over_time:
        df = pd.DataFrame(over_time)
        st.line_chart(df.set_index("date")[["runs"]])

    st.subheader("Recent Runs")
    runs = get_recent_agent_runs(days, limit=200)
    if not runs:
        st.info("No agent runs logged in this window.")
        return
    table = pd.DataFrame(runs)
    st.dataframe(
        table[
            [
                "created_at",
                "question",
                "llm_model",
                "tool_call_count",
                "total_elapsed_ms",
                "total_tokens",
                "status",
                "review_status",
                "selected_patient_display_name",
                "run_id",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    run_ids = [r["run_id"] for r in runs]
    selected = st.selectbox("Inspect a run", options=["—"] + run_ids)
    if selected and selected != "—":
        _render_run_inspector(selected)


def _render_run_inspector(run_id: str) -> None:
    detail = get_agent_run_detail(run_id)
    if not detail:
        st.warning("Run not found.")
        return
    log_agent_run_viewed(st.session_state.get("user_id"), run_id)
    run = detail["run"]
    st.markdown(f"### Run `{run_id}`")
    st.markdown(f"**Question:** {run.get('question')}")
    if run.get("selected_patient_source_id"):
        st.info(
            f"📌 Pinned patient: {run.get('selected_patient_display_name')} "
            f"(`{run.get('selected_patient_source_id')}`, {run.get('selected_patient_selection_origin')})"
        )
    st.caption(
        f"Model: {run.get('llm_model')} · Status: {run.get('status')} · "
        f"Tokens: {run.get('total_tokens')} · {run.get('total_elapsed_ms')} ms"
    )

    for ev in _timeline_rows(detail):
        etype = ev["event_type"]
        payload = json.loads(ev["payload_json"]) if ev.get("payload_json") else {}
        if etype == "thinking_completed":
            with st.expander(f"🤔 Thinking (turn {ev.get('turn_index')})", expanded=False):
                st.markdown(payload.get("text", ""))
        elif etype == "tool_call_started":
            st.markdown(f"**→ calling `{ev.get('tool_name')}`**")
            st.json(payload.get("arguments", {}))
        elif etype == "tool_call_completed":
            with st.expander(f"🔧 {ev.get('tool_name')} result", expanded=False):
                if ev.get("payload_truncated"):
                    st.caption(f"⚠️ truncated — {ev.get('payload_bytes')} bytes, sha256 {ev.get('payload_hash')}")
                result = payload.get("result")
                sql = payload.get("sql_executed")
                if sql:
                    st.code("\n".join(x.get("sql", "") for x in sql), language="sql")
                _render_result(result)
        elif etype == "assistant_text_completed":
            st.markdown(payload.get("text", ""))
        elif etype in ("cap_reached", "run_failed"):
            st.error(f"{etype}: {payload}")

    st.markdown("### Final answer")
    st.success(run.get("final_answer_text") or "(no final text)")

    _render_review_panel(run)


def _render_result(result) -> None:
    """Render a tool result: dataframe when tabular, else JSON."""
    if isinstance(result, dict):
        for key in ("sample", "items", "matches", "rows"):
            if isinstance(result.get(key), list) and result[key]:
                st.dataframe(pd.DataFrame(result[key]), use_container_width=True, hide_index=True)
                break
        st.json(result)
    elif result is not None:
        st.write(result)


def _render_review_panel(run: dict) -> None:
    st.markdown("### QA Review")
    with st.form(f"review-{run['run_id']}"):
        options = ["unreviewed", "verified", "issue", "ignored"]
        current = run.get("review_status", "unreviewed")
        status = st.selectbox(
            "Review status",
            options,
            index=options.index(current) if current in options else 0,
        )
        notes = st.text_area("Notes", value=run.get("review_notes") or "")
        issue_url = st.text_input("Issue URL", value=run.get("issue_url") or "")
        if st.form_submit_button("Save review"):
            set_run_review_status(
                run["run_id"],
                review_status=status,
                reviewed_by=st.session_state.get("user_id"),
                notes=notes,
                issue_url=issue_url or None,
            )
            st.success("Review saved.")
            st.rerun()


def _render_tools_tab(days: int) -> None:
    rows = get_tool_breakdown(days)
    if not rows:
        st.info("No tool calls logged.")
        return
    df = pd.DataFrame(rows)
    df["avg_ms"] = df["avg_ms"].round(0)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.bar_chart(df.set_index("tool_name")[["count"]])


def _render_patient_access_tab(days: int) -> None:
    col1, col2 = st.columns(2)
    source_id = col1.text_input("Filter by source_id")
    user_id_raw = col2.text_input("Filter by user_id")
    user_id = int(user_id_raw) if user_id_raw.strip().isdigit() else None
    rows = get_patient_access(source_id=source_id or None, user_id=user_id, days=days)
    if not rows:
        st.info("No patient-access records for this filter.")
        return
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def main() -> None:
    _guard_admin()
    st.title("🧠 Agentic Analytics")

    cfg = AgentLoggingConfig.from_streamlit()
    if cfg.mode == "full":
        st.warning(
            "⚠️ **Full PHI logging enabled** — tool arguments, results, and "
            "patient identity are stored verbatim. Protect this database like the "
            "clinical source."
        )
    elif cfg.mode == "disabled":
        st.error("Agentic logging is **disabled** in config — no new runs are being recorded.")

    days = st.selectbox("Time window (days)", [1, 7, 30, 90], index=2)
    tabs = st.tabs(["Runs", "Tools", "Patient Access"])
    with tabs[0]:
        _render_runs_tab(days)
    with tabs[1]:
        _render_tools_tab(days)
    with tabs[2]:
        _render_patient_access_tab(days)


if __name__ == "__main__":
    main()
