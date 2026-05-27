"""Read + pin-event write helpers for agentic run logging."""

from __future__ import annotations

from orm.models import AgentPatientAccess, PatientSelectionEvent, SessionLocal
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
