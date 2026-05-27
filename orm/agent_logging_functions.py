"""Read + pin-event write helpers for agentic run logging."""

from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy import case
from sqlalchemy import func as sqla_func

from orm.models import (
    AdminAction,
    AgentPatientAccess,
    AgentRun,
    AgentRunEvent,
    PatientSelectionEvent,
    SessionLocal,
    ToolCall,
)
from utils.quick_logger import get_logger

logger = get_logger(__name__)


def log_patient_selection(
    *,
    session_id: str,
    user_id: int,
    source_id: str | None,
    display_name: str | None,
    selection_origin: str,
    action: str,
    previous_source_id: str | None = None,
    run_id: str | None = None,
) -> None:
    """Record a pin set/clear.

    Always writes PatientSelectionEvent. Writes AgentPatientAccess for both
    selection_chosen and selection_cleared so the Patient Access tab can answer
    who selected or cleared a pinned patient without scanning event JSON.
    """
    try:
        with SessionLocal() as session:
            session.add(
                PatientSelectionEvent(
                    session_id=session_id,
                    run_id=run_id,
                    user_id=user_id,
                    source_id=source_id,
                    previous_source_id=previous_source_id,
                    display_name=display_name,
                    selection_origin=selection_origin,
                    action=action,
                )
            )
            if action == "selected" and source_id:
                session.add(
                    AgentPatientAccess(
                        run_id=run_id,
                        session_id=session_id,
                        user_id=user_id,
                        source_id=source_id,
                        display_name=display_name,
                        access_type="selection_chosen",
                        access_origin=selection_origin,
                    )
                )
            if action == "cleared" and previous_source_id:
                session.add(
                    AgentPatientAccess(
                        run_id=run_id,
                        session_id=session_id,
                        user_id=user_id,
                        source_id=previous_source_id,
                        display_name=None,
                        access_type="selection_cleared",
                        access_origin=selection_origin,
                    )
                )
            session.commit()
    except Exception as e:
        logger.warning("Failed to log patient selection: %s", e)


def get_agent_run_stats(days: int = 30) -> dict:
    since = datetime.now() - timedelta(days=days)
    try:
        with SessionLocal() as s:
            base = s.query(AgentRun).filter(AgentRun.created_at >= since)
            total = base.count() or 0
            successes = base.filter(AgentRun.success == True).count() or 0  # noqa: E712
            capped = base.filter(AgentRun.status == "cap_reached").count() or 0
            avg_tools = (
                s.query(sqla_func.avg(AgentRun.tool_call_count)).filter(AgentRun.created_at >= since).scalar() or 0
            )
            avg_latency = (
                s.query(sqla_func.avg(AgentRun.total_elapsed_ms)).filter(AgentRun.created_at >= since).scalar() or 0
            )
            total_tokens = (
                s.query(sqla_func.sum(AgentRun.total_tokens)).filter(AgentRun.created_at >= since).scalar() or 0
            )
            return {
                "total_runs": total,
                "success_rate": round(successes / total * 100, 1) if total else 0.0,
                "cap_rate": round(capped / total * 100, 1) if total else 0.0,
                "avg_tool_calls": round(float(avg_tools), 2),
                "avg_latency_ms": float(avg_latency),
                "total_tokens": int(total_tokens),
            }
    except Exception as e:
        logger.warning("get_agent_run_stats failed: %s", e)
        return {
            "total_runs": 0,
            "success_rate": 0.0,
            "cap_rate": 0.0,
            "avg_tool_calls": 0.0,
            "avg_latency_ms": 0.0,
            "total_tokens": 0,
        }


def get_recent_agent_runs(days: int = 7, limit: int = 100) -> list[dict]:
    since = datetime.now() - timedelta(days=days)
    try:
        with SessionLocal() as s:
            rows = (
                s.query(AgentRun)
                .filter(AgentRun.created_at >= since)
                .order_by(AgentRun.created_at.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "run_id": r.run_id,
                    "created_at": r.created_at,
                    "user_id": r.user_id,
                    "question": r.question,
                    "llm_model": r.llm_model,
                    "selected_patient_source_id": r.selected_patient_source_id,
                    "selected_patient_display_name": r.selected_patient_display_name,
                    "tool_call_count": r.tool_call_count,
                    "total_elapsed_ms": r.total_elapsed_ms,
                    "total_tokens": r.total_tokens,
                    "status": r.status,
                    "cap_reached": r.cap_reached,
                    "review_status": r.review_status,
                }
                for r in rows
            ]
    except Exception as e:
        logger.warning("get_recent_agent_runs failed: %s", e)
        return []


def get_agent_run_detail(run_id: str) -> dict:
    try:
        with SessionLocal() as s:
            run = s.query(AgentRun).filter_by(run_id=run_id).first()
            if run is None:
                return {}
            events = s.query(AgentRunEvent).filter_by(run_id=run_id).order_by(AgentRunEvent.seq).all()
            tools = s.query(ToolCall).filter_by(run_id=run_id).order_by(ToolCall.call_index).all()
            access = s.query(AgentPatientAccess).filter_by(run_id=run_id).all()
            return {
                "run": {c.name: getattr(run, c.name) for c in run.__table__.columns},
                "events": [{c.name: getattr(e, c.name) for c in e.__table__.columns} for e in events],
                "tool_calls": [{c.name: getattr(t, c.name) for c in t.__table__.columns} for t in tools],
                "patient_access": [{c.name: getattr(a, c.name) for c in a.__table__.columns} for a in access],
            }
    except Exception as e:
        logger.warning("get_agent_run_detail failed: %s", e)
        return {}


def get_tool_breakdown(days: int = 30) -> list[dict]:
    since = datetime.now() - timedelta(days=days)
    try:
        with SessionLocal() as s:
            rows = (
                s.query(
                    ToolCall.tool_name,
                    sqla_func.count(ToolCall.id).label("count"),
                    sqla_func.avg(ToolCall.elapsed_ms).label("avg_ms"),
                    sqla_func.sum(case((ToolCall.success == True, 0), else_=1)).label("failures"),  # noqa: E712
                )
                .filter(ToolCall.created_at >= since, ToolCall.run_id.isnot(None))
                .group_by(ToolCall.tool_name)
                .order_by(sqla_func.count(ToolCall.id).desc())
                .all()
            )
            return [
                {
                    "tool_name": r.tool_name,
                    "count": r.count,
                    "avg_ms": float(r.avg_ms or 0),
                    "failures": int(r.failures or 0),
                }
                for r in rows
            ]
    except Exception as e:
        logger.warning("get_tool_breakdown failed: %s", e)
        return []


def get_patient_access(
    source_id: str | None = None, user_id: int | None = None, days: int = 30, limit: int = 200
) -> list[dict]:
    since = datetime.now() - timedelta(days=days)
    try:
        with SessionLocal() as s:
            q = s.query(AgentPatientAccess).filter(AgentPatientAccess.created_at >= since)
            if source_id:
                q = q.filter(AgentPatientAccess.source_id == source_id)
            if user_id:
                q = q.filter(AgentPatientAccess.user_id == user_id)
            rows = q.order_by(AgentPatientAccess.created_at.desc()).limit(limit).all()
            return [{c.name: getattr(a, c.name) for c in a.__table__.columns} for a in rows]
    except Exception as e:
        logger.warning("get_patient_access failed: %s", e)
        return []


def get_runs_over_time(days: int = 30) -> list[dict]:
    since = datetime.now() - timedelta(days=days)
    try:
        with SessionLocal() as s:
            date_expr = sqla_func.strftime("%Y-%m-%d", AgentRun.created_at)
            rows = (
                s.query(
                    date_expr.label("date"),
                    sqla_func.count(AgentRun.id).label("runs"),
                    sqla_func.avg(AgentRun.total_elapsed_ms).label("avg_latency"),
                )
                .filter(AgentRun.created_at >= since)
                .group_by(date_expr)
                .order_by(date_expr)
                .all()
            )
            return [{"date": r.date, "runs": r.runs, "avg_latency": float(r.avg_latency or 0)} for r in rows]
    except Exception as e:
        logger.warning("get_runs_over_time failed: %s", e)
        return []


def set_run_review_status(
    run_id: str,
    *,
    review_status: str,
    reviewed_by: int | None = None,
    notes: str | None = None,
    issue_url: str | None = None,
) -> bool:
    try:
        with SessionLocal() as s:
            run = s.query(AgentRun).filter_by(run_id=run_id).first()
            if run is None:
                return False
            run.review_status = review_status
            run.reviewed_by = reviewed_by
            run.reviewed_at = datetime.now()
            if notes is not None:
                run.review_notes = notes
            if issue_url is not None:
                run.issue_url = issue_url
            s.commit()
            return True
    except Exception as e:
        logger.warning("set_run_review_status failed: %s", e)
        return False


def log_agent_run_viewed(admin_id: int | None, run_id: str) -> bool:
    """Audit that an admin opened a full-fidelity run trace."""
    if admin_id is None:
        return False
    try:
        with SessionLocal() as s:
            s.add(
                AdminAction(
                    admin_id=admin_id,
                    action_type="agent_run_view",
                    target_entity_type="agent_run",
                    target_entity_id=run_id,
                    description=f"Viewed agentic run trace {run_id}",
                    success=True,
                )
            )
            s.commit()
            return True
    except Exception as e:
        logger.warning("log_agent_run_viewed failed: %s", e)
        return False


def prune_agent_logs(retention_days: int) -> dict:
    """Delete agent logs older than retention_days. Children before parents.
    retention_days <= 0 is a no-op (keep indefinitely). Returns counts deleted."""
    if retention_days <= 0:
        return {"runs": 0, "events": 0, "tool_calls": 0, "patient_access": 0}
    cutoff = datetime.now() - timedelta(days=retention_days)
    counts = {"runs": 0, "events": 0, "tool_calls": 0, "patient_access": 0}
    try:
        with SessionLocal() as s:
            old_ids = [r.run_id for r in s.query(AgentRun.run_id).filter(AgentRun.created_at < cutoff).all()]
            if not old_ids:
                return counts
            counts["events"] = (
                s.query(AgentRunEvent).filter(AgentRunEvent.run_id.in_(old_ids)).delete(synchronize_session=False)
            )
            counts["tool_calls"] = (
                s.query(ToolCall).filter(ToolCall.run_id.in_(old_ids)).delete(synchronize_session=False)
            )
            counts["patient_access"] = (
                s.query(AgentPatientAccess)
                .filter(AgentPatientAccess.run_id.in_(old_ids))
                .delete(synchronize_session=False)
            )
            counts["runs"] = s.query(AgentRun).filter(AgentRun.run_id.in_(old_ids)).delete(synchronize_session=False)
            s.commit()
    except Exception as e:
        logger.warning("prune_agent_logs failed: %s", e)
    return counts
