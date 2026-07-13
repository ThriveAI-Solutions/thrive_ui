"""Admin-authorized evaluation run service.

Owns the durable lifecycle of an :class:`~orm.evaluation_models.EvaluationRun`:
creation (with atomically-inserted ordered case results), synchronous
execution of a single feedback case, admin verdict review, and retention
purging of expired PHI-bearing payloads.

Asynchronous execution, the claim/heartbeat queue, and the process-local
worker live in :mod:`evals.worker`; this module holds the shared, executor-
backed run body those callers reuse plus the pure database operations. Every
mutation re-checks database-backed admin authorization rather than trusting
Streamlit session state.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal, Optional, Sequence

from sqlalchemy.orm import Session, joinedload

from orm.evaluation_models import (
    AdminNotification,
    AgentRunFeedback,
    EvaluationCase,
    EvaluationCaseResult,
    EvaluationReviewEvent,
    EvaluationRun,
)
from orm.models import AgentRun, RoleTypeEnum, SessionLocal, User

FINAL_VERDICTS = ("correct", "incorrect", "cant_tell")

# Result lifecycle: a case begins pending, is set running while executing, and
# ends completed/failed; cancelled marks a pending case skipped after an admin
# cancellation. The run loop executes pending cases plus any "running" case
# left behind by an interrupted worker (recovery resets those to pending), but
# never re-runs a case that already reached a terminal completed/failed/cancelled
# state — that would loop forever on a failing case within one pass.
_RUNNABLE_RESULT_STATUSES = frozenset({"pending", "running"})


class EvaluationServiceError(RuntimeError):
    """Raised for invalid evaluation-run requests (bad input, unknown case)."""


@dataclass(frozen=True)
class EvaluationRunView:
    """Immutable snapshot of run identity/state, safe to hand outside the
    service transaction (never a live SQLAlchemy row)."""

    run_id: str
    run_type: str
    execution_mode: Literal["synchronous", "asynchronous"]
    status: str
    total_cases: int
    completed_cases: int
    failed_cases: int


def _require_admin(session: Session, admin_user_id: int) -> None:
    row = session.query(User).options(joinedload(User.role)).filter(User.id == admin_user_id).one_or_none()
    if row is None or row.role is None or row.role.role != RoleTypeEnum.ADMIN:
        raise PermissionError("Evaluation runs require an admin role.")


def _run_view(run: EvaluationRun) -> EvaluationRunView:
    return EvaluationRunView(
        run_id=run.run_id,
        run_type=run.run_type,
        execution_mode=run.execution_mode,
        status=run.status,
        total_cases=run.total_cases,
        completed_cases=run.completed_cases,
        failed_cases=run.failed_cases,
    )


def _classify_run(session: Session, cases: Sequence[EvaluationCase]) -> tuple[str, str]:
    """Return ``(run_type, execution_mode)`` for a selected set of cases.

    Exactly one feedback case runs synchronously; a feedback batch, any
    curated selection, and the full curated suite run asynchronously.
    """
    source_types = {case.source_type for case in cases}
    if source_types == {"feedback"}:
        if len(cases) == 1:
            return "single_feedback", "synchronous"
        return "feedback_batch", "asynchronous"

    # Any curated case present -> asynchronous. Distinguish the full active
    # curated suite from a curated subset for reporting/labelling.
    if source_types == {"curated"}:
        active_curated = {
            cid
            for (cid,) in session.query(EvaluationCase.id)
            .filter(EvaluationCase.source_type == "curated", EvaluationCase.status == "active")
            .all()
        }
        selected = {case.id for case in cases}
        if active_curated and selected == active_curated:
            return "full_suite", "asynchronous"
        return "curated_subset", "asynchronous"

    # Mixed feedback + curated selection.
    return "curated_subset", "asynchronous"


def create_evaluation_run(case_ids: list[int], admin_user_id: int) -> EvaluationRunView:
    """Create a durable queued run plus one ordered result row per case,
    atomically. Requires a database-backed admin; raises before any insert
    for non-admins or empty/unknown case selections."""
    if not case_ids:
        raise EvaluationServiceError("An evaluation run requires at least one case.")

    with SessionLocal() as session:
        try:
            _require_admin(session, admin_user_id)

            # Preserve caller order for ordinals while loading in one query.
            found = {c.id: c for c in session.query(EvaluationCase).filter(EvaluationCase.id.in_(case_ids)).all()}
            missing = [cid for cid in case_ids if cid not in found]
            if missing:
                raise EvaluationServiceError(f"Unknown evaluation case ids: {missing}")
            cases = [found[cid] for cid in case_ids]

            run_type, execution_mode = _classify_run(session, cases)
            run = EvaluationRun(
                run_id=str(uuid.uuid4()),
                run_type=run_type,
                execution_mode=execution_mode,
                status="queued",
                requested_by=admin_user_id,
                total_cases=len(cases),
                completed_cases=0,
                failed_cases=0,
            )
            session.add(run)
            session.flush()

            for ordinal, case in enumerate(cases):
                session.add(
                    EvaluationCaseResult(
                        evaluation_run_id=run.id,
                        evaluation_case_id=case.id,
                        ordinal=ordinal,
                        attempt=1,
                        status="pending",
                    )
                )
            session.commit()
            return _run_view(run)
        except Exception:
            session.rollback()
            raise


def request_cancellation(run_id: str, admin_user_id: int) -> EvaluationRunView:
    """Flag a queued/running run for cancellation; the worker stops after the
    current case and marks remaining pending cases cancelled."""
    with SessionLocal() as session:
        try:
            _require_admin(session, admin_user_id)
            run = session.query(EvaluationRun).filter(EvaluationRun.run_id == run_id).one_or_none()
            if run is None:
                raise EvaluationServiceError(f"Evaluation run {run_id} not found.")
            if run.status in ("completed", "completed_with_errors", "failed", "cancelled"):
                return _run_view(run)
            run.cancel_requested = True
            session.commit()
            return _run_view(run)
        except Exception:
            session.rollback()
            raise


def resume_run(run_id: str, admin_user_id: int) -> EvaluationRunView:
    """Requeue a run that was interrupted or requeued so the async worker
    re-executes its unfinished cases. No-op for terminal-success runs."""
    with SessionLocal() as session:
        try:
            _require_admin(session, admin_user_id)
            run = session.query(EvaluationRun).filter(EvaluationRun.run_id == run_id).one_or_none()
            if run is None:
                raise EvaluationServiceError(f"Evaluation run {run_id} not found.")
            has_unfinished = (
                session.query(EvaluationCaseResult)
                .filter(
                    EvaluationCaseResult.evaluation_run_id == run.id,
                    EvaluationCaseResult.status.in_(tuple(_RUNNABLE_RESULT_STATUSES)),
                )
                .count()
                > 0
            )
            if has_unfinished and run.status not in ("running",):
                run.status = "queued"
                run.cancel_requested = False
                for result in (
                    session.query(EvaluationCaseResult)
                    .filter(
                        EvaluationCaseResult.evaluation_run_id == run.id,
                        EvaluationCaseResult.status == "running",
                    )
                    .all()
                ):
                    result.status = "pending"
                    result.started_at = None
                session.commit()
            return _run_view(run)
        except Exception:
            session.rollback()
            raise


def record_final_verdict(
    result_id: int,
    admin_user_id: int,
    verdict: Literal["correct", "incorrect", "cant_tell"],
    note: str | None = None,
) -> None:
    """Set an authoritative admin verdict on a case result and append a review
    event with old/new values, in one transaction. Only the three final
    values are accepted; the LLM's triage suggestion is never authoritative."""
    if verdict not in FINAL_VERDICTS:
        raise EvaluationServiceError(f"Final verdict must be one of {FINAL_VERDICTS}.")
    if note is not None and len(note) > 1000:
        raise EvaluationServiceError("Review note must be 1000 characters or fewer.")

    with SessionLocal() as session:
        try:
            _require_admin(session, admin_user_id)
            result = session.query(EvaluationCaseResult).filter(EvaluationCaseResult.id == result_id).one_or_none()
            if result is None:
                raise EvaluationServiceError(f"Evaluation result {result_id} not found.")
            old_verdict = result.final_verdict
            result.final_verdict = verdict
            result.review_note = note
            result.reviewed_by = admin_user_id
            result.reviewed_at = datetime.now(timezone.utc)
            session.add(
                EvaluationReviewEvent(
                    result_id=result.id,
                    reviewer_id=admin_user_id,
                    old_verdict=old_verdict,
                    new_verdict=verdict,
                    note=note,
                )
            )
            session.commit()
        except Exception:
            session.rollback()
            raise


# --------------------------------------------------------------------------- #
# Shared run body (used by execute_synchronous_run and the async worker).
# --------------------------------------------------------------------------- #


def _result_level_judge(turns: tuple[dict, ...]) -> tuple[Optional[str], Optional[str]]:
    """Case-level triage = the last judged turn's suggestion, if any."""
    for turn in reversed(turns):
        judge = turn.get("judge")
        if judge:
            return judge.get("suggestion"), (judge.get("reason") or "")[:500]
    return None, None


def _persist_case_execution(session: Session, run: EvaluationRun, result: EvaluationCaseResult, execution) -> None:
    """Write one CaseExecution outcome onto its result row and recompute run
    counters from the persisted result statuses (resume-safe: no blind
    increment)."""
    result.status = execution.status  # "completed" | "failed"
    result.completed_at = datetime.now(timezone.utc)
    result.result_json = json.dumps({"patient": execution.patient, "turns": list(execution.turns)}, default=str)
    result.error_type = execution.error_type
    result.error_message = execution.error_message
    if execution.status == "completed":
        verdict, reason = _result_level_judge(execution.turns)
        result.judge_verdict = verdict
        result.judge_reason = reason
    session.flush()
    _recount_run(session, run)


def _recount_run(session: Session, run: EvaluationRun) -> None:
    counts: dict[str, int] = {}
    for status, count in (
        session.query(EvaluationCaseResult.status, EvaluationCaseResult.id)
        .filter(EvaluationCaseResult.evaluation_run_id == run.id)
        .all()
    ):
        counts[status] = counts.get(status, 0) + 1
    run.completed_cases = counts.get("completed", 0)
    run.failed_cases = counts.get("failed", 0)


def _finalize_run_status(session: Session, run: EvaluationRun) -> None:
    statuses = [
        s
        for (s,) in session.query(EvaluationCaseResult.status)
        .filter(EvaluationCaseResult.evaluation_run_id == run.id)
        .all()
    ]
    _recount_run(session, run)
    if any(s == "cancelled" for s in statuses):
        run.status = "cancelled"
    elif all(s == "completed" for s in statuses):
        run.status = "completed"
    elif run.completed_cases == 0:
        run.status = "failed"
    else:
        run.status = "completed_with_errors"
    run.completed_at = datetime.now(timezone.utc)


def _notify_completion(session: Session, run: EvaluationRun) -> None:
    session.add(
        AdminNotification(
            user_id=run.requested_by,
            evaluation_run_id=run.id,
            kind="evaluation_completed",
        )
    )


async def run_evaluation_cases(run_id: str, resources, *, heartbeat=None) -> EvaluationRunView:
    """Execute every non-terminal case of a run in order, committing each
    result independently, honoring cancellation between cases, and finalizing
    the run's aggregate status plus a PHI-free completion notification.

    ``resources`` is an :class:`evals.executor.EvaluationResources`. ``heartbeat``
    is an optional ``callable(run_id)`` invoked before and after each case.
    """
    from evals.cases import NormalizedCase
    from evals.executor import execute_case

    while True:
        with SessionLocal() as session:
            run = session.query(EvaluationRun).filter(EvaluationRun.run_id == run_id).one_or_none()
            if run is None:
                raise EvaluationServiceError(f"Evaluation run {run_id} not found.")
            if run.cancel_requested:
                # Mark every remaining runnable case cancelled and stop.
                for result in (
                    session.query(EvaluationCaseResult)
                    .filter(
                        EvaluationCaseResult.evaluation_run_id == run.id,
                        EvaluationCaseResult.status.in_(tuple(_RUNNABLE_RESULT_STATUSES)),
                    )
                    .all()
                ):
                    result.status = "cancelled"
                # Flush so _finalize_run_status re-reads the cancelled statuses;
                # SessionLocal has autoflush disabled.
                session.flush()
                _finalize_run_status(session, run)
                _notify_completion(session, run)
                view = _run_view(run)
                session.commit()
                return view

            result = (
                session.query(EvaluationCaseResult)
                .filter(
                    EvaluationCaseResult.evaluation_run_id == run.id,
                    EvaluationCaseResult.status.in_(tuple(_RUNNABLE_RESULT_STATUSES)),
                )
                .order_by(EvaluationCaseResult.ordinal, EvaluationCaseResult.id)
                .first()
            )
            if result is None:
                _finalize_run_status(session, run)
                _notify_completion(session, run)
                view = _run_view(run)
                session.commit()
                return view

            case_row = session.query(EvaluationCase).filter(EvaluationCase.id == result.evaluation_case_id).one()
            payload = json.loads(case_row.payload_json)
            result.status = "running"
            result.started_at = datetime.now(timezone.utc)
            result_id = result.id
            session.commit()

        if heartbeat is not None:
            heartbeat(run_id)

        # Executor never raises: a failed case is returned as status="failed".
        case = NormalizedCase.from_payload(payload)
        execution = await execute_case(case, resources)

        with SessionLocal() as session:
            run = session.query(EvaluationRun).filter(EvaluationRun.run_id == run_id).one()
            result = session.query(EvaluationCaseResult).filter(EvaluationCaseResult.id == result_id).one()
            _persist_case_execution(session, run, result, execution)
            session.commit()

        if heartbeat is not None:
            heartbeat(run_id)


def execute_synchronous_run(run_id: str, admin_user_id: int, resources=None) -> EvaluationRunView:
    """Run a single-feedback synchronous run to completion in the calling
    thread (Streamlit request). Bypasses the async worker/queue entirely."""
    import asyncio

    with SessionLocal() as session:
        _require_admin(session, admin_user_id)
        run = session.query(EvaluationRun).filter(EvaluationRun.run_id == run_id).one_or_none()
        if run is None:
            raise EvaluationServiceError(f"Evaluation run {run_id} not found.")
        if run.execution_mode != "synchronous":
            raise EvaluationServiceError("execute_synchronous_run only runs synchronous runs.")
        if run.status not in ("queued", "running"):
            return _run_view(run)
        run.status = "running"
        run.started_at = datetime.now(timezone.utc)
        session.commit()

    if resources is None:
        from evals.worker import build_default_resources

        resources = build_default_resources()

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(run_evaluation_cases(run_id, resources))
    finally:
        loop.close()


# --------------------------------------------------------------------------- #
# Recovery + retention.
# --------------------------------------------------------------------------- #


def mark_interrupted_runs(now: datetime, heartbeat_timeout_s: float = 120.0) -> int:
    """Requeue runs whose worker died: any ``running`` run whose heartbeat is
    older than the timeout becomes ``queued`` again and its ``running`` case
    reverts to ``pending`` so resume re-executes only unfinished work.
    Returns the number of runs requeued."""
    from datetime import timedelta

    cutoff = now - timedelta(seconds=heartbeat_timeout_s)
    with SessionLocal() as session:
        try:
            stale = (
                session.query(EvaluationRun)
                .filter(
                    EvaluationRun.status == "running",
                    (EvaluationRun.heartbeat_at.is_(None)) | (EvaluationRun.heartbeat_at < cutoff),
                )
                .all()
            )
            for run in stale:
                run.status = "queued"
                for result in (
                    session.query(EvaluationCaseResult)
                    .filter(
                        EvaluationCaseResult.evaluation_run_id == run.id,
                        EvaluationCaseResult.status == "running",
                    )
                    .all()
                ):
                    result.status = "pending"
                    result.started_at = None
            session.commit()
            return len(stale)
        except Exception:
            session.rollback()
            raise


def purge_expired_evaluation_payloads(now: datetime) -> int:
    """Retention: clear PHI-bearing payloads (case ``payload_json`` and result
    ``result_json``) for expired cases, mark those cases ``expired``, and keep
    only run identity/status/timestamps/counts and audit review events. Never
    deletes review history. Returns the number of cases expired."""
    with SessionLocal() as session:
        try:
            expired_cases = (
                session.query(EvaluationCase)
                .filter(
                    EvaluationCase.expires_at.isnot(None),
                    EvaluationCase.expires_at < now,
                    EvaluationCase.status != "expired",
                )
                .all()
            )
            expired_ids = [c.id for c in expired_cases]
            for case in expired_cases:
                case.status = "expired"
                case.payload_json = json.dumps({"expired": True})
            if expired_ids:
                for result in (
                    session.query(EvaluationCaseResult)
                    .filter(EvaluationCaseResult.evaluation_case_id.in_(expired_ids))
                    .all()
                ):
                    result.result_json = None
            session.commit()
            return len(expired_ids)
        except Exception:
            session.rollback()
            raise


# --------------------------------------------------------------------------- #
# Authenticated report, notifications, and launch catalog (Task 7/8).
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class NotificationView:
    """PHI-free notification payload — never carries patient, question, answer,
    or tool evidence, only run identity, status, counts, and timestamps."""

    notification_id: int
    run_id: str
    kind: str
    status: str
    total_cases: int
    completed_cases: int
    failed_cases: int
    created_at: Optional[datetime]
    read_at: Optional[datetime]

    @property
    def text(self) -> str:
        return f"Evaluation {self.run_id} completed: {self.completed_cases}/{self.total_cases} cases."


def list_admin_notifications(admin_user_id: int, unread_only: bool = True) -> list[NotificationView]:
    """Return the admin's notifications as PHI-free views, newest first."""
    with SessionLocal() as session:
        _require_admin(session, admin_user_id)
        query = (
            session.query(AdminNotification, EvaluationRun)
            .join(EvaluationRun, AdminNotification.evaluation_run_id == EvaluationRun.id)
            .filter(AdminNotification.user_id == admin_user_id)
        )
        if unread_only:
            query = query.filter(AdminNotification.read_at.is_(None))
        rows = query.order_by(AdminNotification.created_at.desc(), AdminNotification.id.desc()).all()
        return [
            NotificationView(
                notification_id=note.id,
                run_id=run.run_id,
                kind=note.kind,
                status=run.status,
                total_cases=run.total_cases,
                completed_cases=run.completed_cases,
                failed_cases=run.failed_cases,
                created_at=note.created_at,
                read_at=note.read_at,
            )
            for note, run in rows
        ]


def mark_notification_read(notification_id: int, admin_user_id: int) -> None:
    """Mark one of the admin's own notifications read (idempotent)."""
    with SessionLocal() as session:
        try:
            _require_admin(session, admin_user_id)
            note = (
                session.query(AdminNotification)
                .filter(
                    AdminNotification.id == notification_id,
                    AdminNotification.user_id == admin_user_id,
                )
                .one_or_none()
            )
            if note is None:
                raise EvaluationServiceError("Notification not found for this admin.")
            if note.read_at is None:
                note.read_at = datetime.now(timezone.utc)
                session.commit()
        except Exception:
            session.rollback()
            raise


def get_evaluation_run_report(run_id: str, actor_user_id: int) -> dict:
    """Return the full authenticated report for a run. Admin-gated: a non-admin
    is rejected before any PHI-bearing payload is read. Returns plain
    dictionaries (no attached SQLAlchemy rows)."""
    with SessionLocal() as session:
        _require_admin(session, actor_user_id)
        run = session.query(EvaluationRun).filter(EvaluationRun.run_id == run_id).one_or_none()
        if run is None:
            raise EvaluationServiceError(f"Evaluation run {run_id} not found.")

        results = (
            session.query(EvaluationCaseResult, EvaluationCase)
            .join(EvaluationCase, EvaluationCaseResult.evaluation_case_id == EvaluationCase.id)
            .filter(EvaluationCaseResult.evaluation_run_id == run.id)
            .order_by(EvaluationCaseResult.ordinal, EvaluationCaseResult.id)
            .all()
        )

        case_reports = []
        for result, case in results:
            case_payload = {}
            if case.status != "expired" and case.payload_json:
                try:
                    case_payload = json.loads(case.payload_json)
                except (TypeError, ValueError):
                    case_payload = {}
            rerun = {}
            if result.result_json:
                try:
                    rerun = json.loads(result.result_json)
                except (TypeError, ValueError):
                    rerun = {}
            case_reports.append(
                {
                    "result_id": result.id,
                    "ordinal": result.ordinal,
                    "case_id": case.case_id,
                    "source_type": case.source_type,
                    "status": result.status,
                    "expired": case.status == "expired",
                    "concern": case_payload.get("reviewer_guidance", ""),
                    "title": case_payload.get("title", ""),
                    "original_answer": case_payload.get("original_answer"),
                    "original_evidence": case_payload.get("original_evidence", []),
                    "rerun_turns": rerun.get("turns", []),
                    "rerun_patient": rerun.get("patient", {}),
                    "error_type": result.error_type,
                    "error_message": result.error_message,
                    "judge_verdict": result.judge_verdict,
                    "judge_reason": result.judge_reason,
                    "final_verdict": result.final_verdict,
                    "review_note": result.review_note,
                }
            )

        return {
            "run_id": run.run_id,
            "run_type": run.run_type,
            "execution_mode": run.execution_mode,
            "status": run.status,
            "total_cases": run.total_cases,
            "completed_cases": run.completed_cases,
            "failed_cases": run.failed_cases,
            "cancel_requested": run.cancel_requested,
            "cases": case_reports,
        }


def list_feedback_candidates(admin_user_id: int, days: Optional[int] = None) -> list[dict]:
    """List thumbs-down agent feedback available to snapshot into cases. PHI
    (question, patient) is included but stays inside the admin-gated page."""
    from datetime import timedelta

    with SessionLocal() as session:
        _require_admin(session, admin_user_id)
        query = (
            session.query(AgentRunFeedback, AgentRun, User)
            .join(AgentRun, AgentRunFeedback.agent_run_id == AgentRun.id)
            .join(User, AgentRunFeedback.user_id == User.id)
            .filter(AgentRunFeedback.rating == "down")
        )
        if days is not None:
            query = query.filter(AgentRunFeedback.created_at >= datetime.now(timezone.utc) - timedelta(days=days))
        rows = query.order_by(AgentRunFeedback.created_at.desc()).all()
        return [
            {
                "feedback_id": fb.id,
                "username": user.username,
                "organization": user.organization,
                "category": fb.category,
                "comment": fb.comment,
                "created_at": fb.created_at,
                "question": run.question,
                "patient": run.selected_patient_display_name,
                "logging_mode": run.logging_mode,
                # A snapshot needs a completed, non-disabled run to replay.
                "replayable": bool(
                    run.logging_mode != "disabled" and run.selected_patient_source_id and run.final_answer_text
                ),
            }
            for fb, run, user in rows
        ]


def list_curated_cases(admin_user_id: int, include_drafts: bool = False) -> list[dict]:
    """List curated evaluation cases (active by default; drafts optional)."""
    with SessionLocal() as session:
        _require_admin(session, admin_user_id)
        statuses = ("active", "draft") if include_drafts else ("active",)
        rows = (
            session.query(EvaluationCase)
            .filter(EvaluationCase.source_type == "curated", EvaluationCase.status.in_(statuses))
            .order_by(EvaluationCase.created_at.desc())
            .all()
        )
        out = []
        for case in rows:
            title = ""
            try:
                title = json.loads(case.payload_json).get("title", "")
            except (TypeError, ValueError):
                pass
            out.append({"id": case.id, "case_id": case.case_id, "status": case.status, "title": title})
        return out


def activate_curated_case(case_id: int, admin_user_id: int) -> None:
    """Promote a draft curated case to ``active`` so it joins the runnable suite.

    Idempotent for already-active cases; raises if the case is missing, not
    curated, or in a non-activatable status (e.g. ``expired``)."""
    with SessionLocal() as session:
        try:
            _require_admin(session, admin_user_id)
            case = (
                session.query(EvaluationCase)
                .filter(
                    EvaluationCase.id == case_id,
                    EvaluationCase.source_type == "curated",
                )
                .one_or_none()
            )
            if case is None:
                raise EvaluationServiceError("Curated case not found.")
            if case.status == "active":
                return
            if case.status != "draft":
                raise EvaluationServiceError(f"Only draft curated cases can be activated (status: {case.status}).")
            case.status = "active"
            session.commit()
        except Exception:
            session.rollback()
            raise


def launch_evaluation(case_ids: list[int], admin_user_id: int, resources=None) -> EvaluationRunView:
    """Create a run and, for a single feedback case, execute it synchronously
    in-request; every other selection is queued for the async worker. Returns
    the resulting run view."""
    view = create_evaluation_run(case_ids, admin_user_id)
    if view.execution_mode == "synchronous":
        return execute_synchronous_run(view.run_id, admin_user_id, resources=resources)
    return view
