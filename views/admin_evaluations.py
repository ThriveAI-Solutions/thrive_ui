"""Admin Evaluations workspace (Task 7 + 8).

Authenticated catalog + launcher for feedback-derived and curated agent
evaluations, plus the durable report with LLM triage, authoritative admin
verdicts, promotion to curated cases, notifications, and resume/cancel.

All PHI stays inside this admin-gated page. Every service call re-checks
database-backed admin authorization, so the page guard is defense-in-depth,
not the only gate.
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from evals.cases import CuratedCaseDraft, SnapshotUnavailable, promote_feedback_case, snapshot_feedback_case
from orm.evaluation_functions import (
    EvaluationServiceError,
    activate_curated_case,
    get_evaluation_run_report,
    launch_evaluation,
    list_admin_notifications,
    list_curated_cases,
    list_feedback_candidates,
    mark_notification_read,
    record_final_verdict,
    request_cancellation,
    resume_run,
)
from orm.models import RoleTypeEnum
from utils.quick_logger import get_logger

logger = get_logger(__name__)

_VERDICT_LABELS = {"correct": "Correct", "incorrect": "Incorrect", "cant_tell": "Can't tell"}


def _guard_admin() -> None:
    if st.session_state.get("user_role") != RoleTypeEnum.ADMIN.value:
        st.error("You don't have permission to view this page.")
        st.stop()


def _admin_id() -> int:
    """Resolve the logged-in admin's user id.

    The local-auth flow stores the id in the cookie manager as a JSON string
    (``orm.functions.verify_user_credentials``), not directly in session_state,
    so reading only ``session_state['user_id']`` always yielded 0 and every
    service call failed database-backed admin authorization. Mirror the
    resolution the rest of the app uses (see ``views.agent_feedback``): prefer a
    direct session_state value, then fall back to the JSON-encoded cookie.
    """
    direct = st.session_state.get("user_id")
    if direct is not None:
        try:
            return int(direct)
        except (TypeError, ValueError):
            pass
    cookies = st.session_state.get("cookies")
    raw = cookies.get("user_id") if cookies is not None else None
    if raw is not None:
        try:
            return int(json.loads(raw))
        except (TypeError, ValueError, json.JSONDecodeError):
            try:
                return int(raw)
            except (TypeError, ValueError):
                pass
    return 0


# --------------------------------------------------------------------------- #
# Launch dispatch (kept out of render() so it is unit-testable).
# --------------------------------------------------------------------------- #


def _snapshot_feedback_ids(feedback_ids: list[int], admin_id: int) -> list[int]:
    """Snapshot each selected thumbs-down feedback into an immutable case,
    returning the new case ids. Surfaces SnapshotUnavailable per feedback item
    without exposing another user's content."""
    case_ids: list[int] = []
    for fid in feedback_ids:
        try:
            case = snapshot_feedback_case(fid, admin_id)
            case_ids.append(case.id)
        except SnapshotUnavailable as exc:
            st.warning(f"Feedback #{fid} can't be evaluated: {exc}")
    return case_ids


def _launch(case_ids: list[int], admin_id: int) -> None:
    """Create + dispatch a run for the given case ids and route the UI to the
    report (synchronous) or a PHI-free toast (asynchronous)."""
    if not case_ids:
        st.warning("Nothing to launch — no replayable cases were selected.")
        return
    try:
        if len(case_ids) == 1:
            with st.status("Running evaluation…", expanded=False):
                view = launch_evaluation(case_ids, admin_id)
        else:
            view = launch_evaluation(case_ids, admin_id)
    except EvaluationServiceError as exc:
        st.error(str(exc))
        return

    if view.execution_mode == "synchronous":
        st.session_state["evaluation_run_id"] = view.run_id
        st.rerun()
    else:
        st.toast(f"Queued evaluation {view.run_id} ({view.status}); {view.total_cases} cases.")


# --------------------------------------------------------------------------- #
# Rendering.
# --------------------------------------------------------------------------- #


def render(days_int: int) -> None:
    _guard_admin()
    admin_id = _admin_id()

    _render_notifications(admin_id)

    run_id = st.session_state.get("evaluation_run_id")
    if run_id:
        _render_report(admin_id, run_id)
        return

    _render_launch_catalog(admin_id, days_int)


def _render_notifications(admin_id: int) -> None:
    notes = list_admin_notifications(admin_id, unread_only=True)
    if not notes:
        return
    with st.container(border=True):
        st.caption(f"🔔 {len(notes)} unread evaluation notification(s)")
        for note in notes:
            cols = st.columns([0.7, 0.15, 0.15])
            cols[0].write(note.text)
            if cols[1].button("View report", key=f"note_view_{note.notification_id}"):
                st.session_state["evaluation_run_id"] = note.run_id
                st.rerun()
            if cols[2].button("Mark read", key=f"note_read_{note.notification_id}"):
                mark_notification_read(note.notification_id, admin_id)
                st.rerun()


def _render_launch_catalog(admin_id: int, days_int: int) -> None:
    st.subheader("Launch an evaluation")
    source = st.segmented_control(
        "Source",
        options=["Thumbs-down feedback", "Curated suite"],
        selection_mode="single",
        default="Thumbs-down feedback",
        key="eval_source",
    )
    if source == "Curated suite":
        _render_curated_source(admin_id)
    else:
        _render_feedback_source(admin_id, days_int)


def _render_feedback_source(admin_id: int, days_int: int) -> None:
    candidates = list_feedback_candidates(admin_id, days=days_int)
    if not candidates:
        st.info("No thumbs-down agent feedback in this time range.")
        return

    df = pd.DataFrame(candidates)
    display_cols = ["username", "organization", "patient", "category", "question", "created_at", "replayable"]
    display = df[[c for c in display_cols if c in df.columns]]
    event = st.dataframe(
        display,
        hide_index=True,
        on_select="rerun",
        selection_mode="multi-row",
        key="feedback_catalog",
    )
    selected_rows = event.selection.rows if event and event.selection else []
    selected = df.iloc[selected_rows] if selected_rows else df.iloc[[]]
    replayable = selected[selected["replayable"]] if not selected.empty else selected
    feedback_ids = [int(x) for x in replayable["feedback_id"].tolist()]

    mode = "synchronous" if len(feedback_ids) == 1 else "asynchronous"
    with st.container(border=True):
        st.write(f"**{len(feedback_ids)}** replayable case(s) selected — **{mode}** run.")
        if len(selected) != len(replayable):
            st.caption("Feedback whose run had logging disabled or no resolved patient can't be exactly replayed.")
        if st.button("Launch evaluation", type="primary", disabled=not feedback_ids, key="launch_feedback"):
            case_ids = _snapshot_feedback_ids(feedback_ids, admin_id)
            _launch(case_ids, admin_id)


def _render_curated_drafts(admin_id: int, drafts: list[dict]) -> None:
    """Show promoted-but-inactive curated drafts, each with an activation button.

    Activating flips the draft to ``active`` so it joins the runnable suite."""
    if not drafts:
        return
    with st.expander(f"Draft curated cases ({len(drafts)}) — review, then activate", expanded=False):
        st.caption("Promoted feedback cases are saved as drafts. Activate one to add it to the runnable suite.")
        for draft in drafts:
            cols = st.columns([0.8, 0.2])
            cols[0].write(draft["title"] or draft["case_id"])
            if cols[1].button("Activate", key=f"activate_draft_{draft['id']}"):
                try:
                    activate_curated_case(draft["id"], admin_id)
                except EvaluationServiceError as exc:
                    st.error(str(exc))
                else:
                    st.toast("Curated case activated.")
                    st.rerun()


def _render_curated_source(admin_id: int) -> None:
    all_cases = list_curated_cases(admin_id, include_drafts=True)
    drafts = [c for c in all_cases if c["status"] == "draft"]
    cases = [c for c in all_cases if c["status"] == "active"]

    _render_curated_drafts(admin_id, drafts)

    if not cases:
        st.info("No active curated evaluation cases. Promote a reviewed feedback case, then activate the draft.")
        return

    df = pd.DataFrame(cases)
    full_suite = st.checkbox("Run the full active curated suite", key="curated_full_suite")
    if full_suite:
        selected_ids = [int(x) for x in df["id"].tolist()]
        st.caption(f"Full suite: {len(selected_ids)} case(s).")
    else:
        event = st.dataframe(
            df[["case_id", "title", "status"]],
            hide_index=True,
            on_select="rerun",
            selection_mode="multi-row",
            key="curated_catalog",
        )
        rows = event.selection.rows if event and event.selection else []
        selected_ids = [int(df.iloc[r]["id"]) for r in rows]

    with st.container(border=True):
        st.write(f"**{len(selected_ids)}** curated case(s) selected — **asynchronous** run.")
        if st.button("Launch evaluation", type="primary", disabled=not selected_ids, key="launch_curated"):
            _launch(selected_ids, admin_id)


def _render_report(admin_id: int, run_id: str) -> None:
    try:
        report = get_evaluation_run_report(run_id, admin_id)
    except EvaluationServiceError as exc:
        st.error(str(exc))
        if st.button("← Back to catalog"):
            st.session_state.pop("evaluation_run_id", None)
            st.rerun()
        return

    header = st.columns([0.6, 0.2, 0.2])
    header[0].subheader(f"Evaluation {report['run_id']}")
    header[0].caption(
        f"{report['run_type']} · {report['execution_mode']} · status: **{report['status']}** · "
        f"{report['completed_cases']}/{report['total_cases']} done, {report['failed_cases']} failed"
    )
    if header[1].button("← Back"):
        st.session_state.pop("evaluation_run_id", None)
        st.rerun()

    if report["status"] in ("queued", "running") and not report["cancel_requested"]:
        if header[2].button("Cancel run"):
            request_cancellation(run_id, admin_id)
            st.rerun()
    if report["status"] in ("queued", "interrupted", "failed", "completed_with_errors", "cancelled"):
        if header[2].button("Resume unfinished"):
            resume_run(run_id, admin_id)
            st.rerun()

    for case in report["cases"]:
        _render_case_report(admin_id, case)


def _render_case_report(admin_id: int, case: dict) -> None:
    with st.container(border=True):
        st.markdown(f"**Case {case['ordinal'] + 1}** · `{case['case_id']}` · status: {case['status']}")
        if case["expired"]:
            st.info("This case's PHI payload has been purged by retention; audit metadata is retained.")
        if case["concern"]:
            st.caption(f"Reviewer concern: {case['concern']}")

        if case["status"] == "failed":
            st.error(f"Execution failed: {case['error_type']}: {case['error_message']}")

        left, right = st.columns(2)
        with left:
            st.markdown("**Original answer**")
            st.write(case["original_answer"] or "—")
        with right:
            st.markdown("**Re-run answer**")
            turns = case["rerun_turns"]
            if turns:
                for turn in turns:
                    st.write(turn.get("answer") or "—")
            else:
                st.write("—")

        if case["judge_verdict"]:
            st.caption(f"LLM triage (not authoritative): {case['judge_verdict']} — {case['judge_reason'] or ''}")

        _render_verdict_controls(admin_id, case)


def _render_verdict_controls(admin_id: int, case: dict) -> None:
    keys = list(_VERDICT_LABELS.keys())
    current = case["final_verdict"]
    cols = st.columns([0.5, 0.5])
    with cols[0]:
        verdict = st.radio(
            "Admin verdict",
            options=keys,
            index=keys.index(current) if current in keys else None,
            format_func=lambda k: _VERDICT_LABELS[k],
            key=f"verdict_{case['result_id']}",
            horizontal=True,
        )
        note = st.text_input("Note (optional)", value=case["review_note"] or "", key=f"note_{case['result_id']}")
        if st.button("Save verdict", key=f"save_{case['result_id']}", disabled=verdict is None):
            record_final_verdict(case["result_id"], admin_id, verdict, note or None)
            st.toast("Verdict saved.")
            st.rerun()
    with cols[1]:
        if case["source_type"] == "feedback" and not case["expired"]:
            if st.button("Promote to curated case", key=f"promote_{case['result_id']}"):
                st.session_state["promote_case_id"] = case["case_id"]
            if st.session_state.get("promote_case_id") == case["case_id"]:
                _promote_dialog(admin_id, case["case_id"])


@st.dialog("Promote to curated case")
def _promote_dialog(admin_id: int, case_id: str) -> None:
    st.caption("Generalize this feedback case into a reusable curated case (saved as a draft).")
    prompt = st.text_area("Reusable prompt", key="promote_prompt")
    followups_raw = st.text_area("Follow-up prompts (one per line)", key="promote_followups")
    patient_req = st.text_input("Patient selection requirement", key="promote_patient_req")
    expected = st.text_area("Expected behavior", key="promote_expected")
    guidance = st.text_area("Reviewer guidance", key="promote_guidance")
    if st.button("Save draft", type="primary"):
        followups = tuple(line.strip() for line in followups_raw.splitlines() if line.strip())
        try:
            promote_feedback_case(
                case_id,
                admin_id,
                CuratedCaseDraft(
                    prompt=prompt,
                    followups=followups,
                    patient_requirements=patient_req,
                    expected_behavior=expected,
                    reviewer_guidance=guidance,
                ),
            )
        except (ValueError, SnapshotUnavailable) as exc:
            st.error(str(exc))
            return
        st.session_state.pop("promote_case_id", None)
        st.toast("Curated draft saved.")
        st.rerun()
