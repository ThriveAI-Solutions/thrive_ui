"""Source-neutral evaluation cases.

Curated cases come from `evals/matrix.py`'s parameterized YAML + roster
conversations. Feedback cases are immutable snapshots of one thumbs-down
`AgentRun` interaction, captured once so later message edits, feedback
changes, or log retention cannot silently change what a re-run replays.

Both shapes normalize into `NormalizedCase` so `evals/executor.py` can run
either kind through one loop.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Literal

from sqlalchemy.orm import joinedload

from evals.matrix import PlannedConversation
from orm.evaluation_models import AgentRunFeedback, EvaluationCase
from orm.models import AgentRun, AgentRunEvent, RoleTypeEnum, SessionLocal, User

SCHEMA_VERSION = 1


class SnapshotUnavailable(RuntimeError):
    """Raised when a feedback interaction cannot be exactly replayed."""


@dataclass(frozen=True)
class NormalizedTurn:
    role: Literal["main", "followup"]
    prompt: str


@dataclass(frozen=True)
class NormalizedCase:
    """One conversation ready for execution, regardless of source."""

    case_id: str
    version: int
    source_type: Literal["feedback", "curated"]
    title: str
    patient_source_id: str
    patient_label: str
    turns: tuple[NormalizedTurn, ...]
    message_history: tuple[dict, ...] = ()
    original_answer: str | None = None
    original_evidence: tuple[dict, ...] = ()
    reviewer_guidance: str = ""

    def to_payload(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "case_id": self.case_id,
            "version": self.version,
            "source_type": self.source_type,
            "title": self.title,
            "patient_source_id": self.patient_source_id,
            "patient_label": self.patient_label,
            "turns": [{"role": t.role, "prompt": t.prompt} for t in self.turns],
            "message_history": list(self.message_history),
            "original_answer": self.original_answer,
            "original_evidence": list(self.original_evidence),
            "reviewer_guidance": self.reviewer_guidance,
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "NormalizedCase":
        return cls(
            case_id=payload["case_id"],
            version=payload["version"],
            source_type=payload["source_type"],
            title=payload["title"],
            patient_source_id=payload["patient_source_id"],
            patient_label=payload.get("patient_label", ""),
            turns=tuple(NormalizedTurn(role=t["role"], prompt=t["prompt"]) for t in payload["turns"]),
            message_history=tuple(payload.get("message_history") or ()),
            original_answer=payload.get("original_answer"),
            original_evidence=tuple(payload.get("original_evidence") or ()),
            reviewer_guidance=payload.get("reviewer_guidance", ""),
        )


@dataclass(frozen=True)
class CuratedCaseDraft:
    """Admin-supplied generalization of a feedback case into a reusable one."""

    prompt: str
    followups: tuple[str, ...]
    patient_requirements: str
    expected_behavior: str
    reviewer_guidance: str


def normalize_curated_conversation(planned: PlannedConversation) -> NormalizedCase:
    """Wrap one `evals.matrix.PlannedConversation` as a `NormalizedCase`."""
    return NormalizedCase(
        case_id=planned.conversation_id,
        version=1,
        source_type="curated",
        title=planned.question_title,
        patient_source_id=planned.source_id,
        patient_label=planned.patient_label,
        turns=tuple(NormalizedTurn(role=t.role, prompt=t.prompt) for t in planned.turns),
        reviewer_guidance=planned.reviewer_note,
    )


def execution_mode(cases: Iterable) -> Literal["synchronous", "asynchronous"]:
    """Exactly one feedback case runs synchronously; everything else — a
    feedback batch, any curated selection, or the full suite — runs
    asynchronously."""
    cases = list(cases)
    if len(cases) == 1 and cases[0].source_type == "feedback":
        return "synchronous"
    return "asynchronous"


def _require_admin(session, admin_user_id: int) -> None:
    row = session.query(User).options(joinedload(User.role)).filter(User.id == admin_user_id).one_or_none()
    if row is None or row.role is None or row.role.role != RoleTypeEnum.ADMIN:
        raise PermissionError("Evaluation snapshot and promotion require an admin role.")


def _extract_tool_evidence(session, run: AgentRun) -> list[dict]:
    """Reconstruct tool evidence from the append-only event timeline.

    In `scrubbed` mode `payload_json` is never written (see
    `agent.run_logger.AgentRunLogger.log_tool_completed`), so evidence here
    naturally carries only the PHI-safe `payload_summary` — full result rows
    stay excluded without extra branching.
    """
    events = (
        session.query(AgentRunEvent)
        .filter(AgentRunEvent.run_id == run.run_id, AgentRunEvent.event_type == "tool_call_completed")
        .order_by(AgentRunEvent.seq)
        .all()
    )
    evidence = []
    for event in events:
        item = {"tool_name": event.tool_name, "result_summary": event.payload_summary}
        if event.payload_json:
            try:
                parsed = json.loads(event.payload_json)
            except (TypeError, ValueError):
                parsed = None
            # `log_tool_completed` writes payload_json as
            # {"result": <rows>, "sql_executed": [...]} in full mode; unwrap the
            # result rows and carry SQL alongside rather than nesting the envelope.
            if isinstance(parsed, dict) and "result" in parsed:
                item["result"] = parsed["result"]
                if parsed.get("sql_executed"):
                    item["sql_executed"] = parsed["sql_executed"]
            elif parsed is not None:
                item["result"] = parsed
        evidence.append(item)
    return evidence


def snapshot_feedback_case(feedback_id: int, admin_user_id: int) -> EvaluationCase:
    """Convert one thumbs-down interaction into an immutable `EvaluationCase`.

    Fails closed: any missing required context (patient, question, final
    answer) or disabled logging raises `SnapshotUnavailable` rather than
    reconstructing a partial replay.
    """
    from agent.logging_config import AgentLoggingConfig

    with SessionLocal() as session:
        _require_admin(session, admin_user_id)

        feedback = session.query(AgentRunFeedback).filter_by(id=feedback_id).one_or_none()
        if feedback is None:
            raise SnapshotUnavailable(f"Feedback {feedback_id} not found.")
        if feedback.rating != "down":
            raise SnapshotUnavailable("Only thumbs-down feedback can be snapshotted.")

        run = session.query(AgentRun).filter_by(id=feedback.agent_run_id).one_or_none()
        if run is None:
            raise SnapshotUnavailable("The originating agent run no longer exists.")
        if run.logging_mode == "disabled":
            raise SnapshotUnavailable("Exact replay is unavailable because agent logging is disabled.")
        if not run.selected_patient_source_id:
            raise SnapshotUnavailable("The originating run has no resolved patient source_id.")
        if not run.question:
            raise SnapshotUnavailable("The originating run has no recorded question.")
        if not run.final_answer_text:
            raise SnapshotUnavailable("The originating run has no final answer to replay.")

        message_history: list = []
        if run.message_history_json:
            try:
                message_history = json.loads(run.message_history_json)
            except (TypeError, ValueError):
                message_history = []

        reviewer_guidance = feedback.comment or ""
        if feedback.category:
            reviewer_guidance = f"[{feedback.category}] {reviewer_guidance}".strip()

        normalized = NormalizedCase(
            case_id=str(uuid.uuid4()),
            version=1,
            source_type="feedback",
            title=run.question[:120],
            patient_source_id=run.selected_patient_source_id,
            patient_label=run.selected_patient_display_name or "",
            turns=(NormalizedTurn(role="main", prompt=run.question),),
            message_history=tuple(message_history),
            original_answer=run.final_answer_text,
            original_evidence=tuple(_extract_tool_evidence(session, run)),
            reviewer_guidance=reviewer_guidance,
        )

        logging_config = AgentLoggingConfig.from_streamlit()
        expires_at = None
        if logging_config.retention_days > 0:
            expires_at = datetime.now(timezone.utc) + timedelta(days=logging_config.retention_days)

        case = EvaluationCase(
            case_id=normalized.case_id,
            version=1,
            source_type="feedback",
            status="active",
            source_feedback_id=feedback.id,
            source_message_id=run.final_message_id,
            source_agent_run_id=run.id,
            payload_json=json.dumps(normalized.to_payload()),
            created_by=admin_user_id,
            expires_at=expires_at,
        )
        session.add(case)
        session.commit()
        session.refresh(case)
        return case


def promote_feedback_case(case_id: str, admin_user_id: int, definition: CuratedCaseDraft) -> EvaluationCase:
    """Create a draft curated case from a feedback snapshot.

    Never opens `evals/questions.yaml` — promotion produces an application
    draft only; exporting an approved draft into the repository suite is a
    separate, controlled development workflow.
    """
    if not definition.prompt.strip():
        raise ValueError("promote_feedback_case requires a non-empty reusable prompt")
    if not definition.patient_requirements.strip():
        raise ValueError("promote_feedback_case requires patient-selection requirements")
    if not definition.expected_behavior.strip():
        raise ValueError("promote_feedback_case requires expected behavior")
    if not definition.reviewer_guidance.strip():
        raise ValueError("promote_feedback_case requires reviewer guidance")

    with SessionLocal() as session:
        _require_admin(session, admin_user_id)

        source = (
            session.query(EvaluationCase)
            .filter_by(case_id=case_id)
            .order_by(EvaluationCase.version.desc())
            .first()
        )
        if source is None:
            raise SnapshotUnavailable(f"Evaluation case {case_id} not found.")

        draft_case_id = str(uuid.uuid4())
        draft_payload = {
            "schema_version": SCHEMA_VERSION,
            "case_id": draft_case_id,
            "version": 1,
            "source_type": "curated",
            "title": definition.prompt[:120],
            "patient_source_id": "",
            "patient_label": "",
            "turns": (
                [{"role": "main", "prompt": definition.prompt}]
                + [{"role": "followup", "prompt": f} for f in definition.followups]
            ),
            "message_history": [],
            "original_answer": None,
            "original_evidence": [],
            "reviewer_guidance": definition.reviewer_guidance,
            "patient_requirements": definition.patient_requirements,
            "expected_behavior": definition.expected_behavior,
        }

        draft = EvaluationCase(
            case_id=draft_case_id,
            version=1,
            source_type="curated",
            status="draft",
            promoted_from_case_id=source.id,
            payload_json=json.dumps(draft_payload),
            created_by=admin_user_id,
        )
        session.add(draft)
        session.commit()
        session.refresh(draft)
        return draft
