"""Reusable, source-neutral evaluation executor.

Runs one `evals.cases.NormalizedCase` through the agentic runner, event
collector, latency attribution, and LLM judge — independent of whether the
case is a curated YAML conversation or a feedback snapshot, and independent
of whether the caller is the developer CLI or (eventually) the Admin
evaluation launcher.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional, Sequence

from evals.cases import NormalizedCase
from evals.collect import run_turn
from evals.judge import judge_turn
from evals.patients import resolve_patient


@dataclass
class EvaluationResources:
    """Already-constructed collaborators one case execution needs."""

    runner: Any
    adapter: Any
    rag: Any
    judge: Optional[Any]


@dataclass(frozen=True)
class CaseExecution:
    """One case's plain, serializable execution result."""

    case_id: str
    status: Literal["completed", "failed"]
    patient: dict
    turns: tuple[dict, ...]
    error_type: Optional[str] = None
    error_message: Optional[str] = None


def _build_deps(adapter, rag, selected_patient, session_id: str):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from agent.deps import AgentDeps
    from orm.models import RoleTypeEnum

    return AgentDeps(
        user_id=0,
        user_role=RoleTypeEnum.DOCTOR,
        session_id=session_id,
        selected_patient=selected_patient,
        last_dataframe=None,
        last_sql=None,
        last_query_meta=None,
        analytics_db=adapter,
        rag=rag,
        sqlite_session=sessionmaker(bind=create_engine("sqlite:///:memory:"))(),
        run_logger=None,
    )


async def execute_case(case: NormalizedCase, resources: EvaluationResources) -> CaseExecution:
    """Resolve the case's exact patient source_id, run every turn, and judge
    each answer. Patient resolution failure fails the case closed — it never
    substitutes a different patient."""
    patient = {"source_id": case.patient_source_id, "display_name": "", "label": case.patient_label}
    try:
        selected = resolve_patient(resources.adapter, case.patient_source_id)
        # Clears the lookup query AND any stale entries left if a prior
        # case errored mid-turn, so turn 1 attribution starts clean.
        resources.adapter.pop_sql_log()
        patient["display_name"] = selected.display_name
        deps = _build_deps(resources.adapter, resources.rag, selected, session_id=f"case-{case.case_id}")

        turns: list[dict] = []
        message_history = list(case.message_history) or None
        for index, turn_spec in enumerate(case.turns):
            turn, all_messages = await run_turn(
                resources.runner, deps, turn_spec.prompt, message_history=message_history
            )
            turn["index"] = index
            turn["role"] = turn_spec.role
            if resources.judge is not None:
                summaries = [
                    f"{tc['tool_name']}: {tc.get('result_summary', '')}"
                    for tc in turn["tool_calls"]
                    if tc.get("completed")
                ]
                turn["judge"] = await judge_turn(
                    resources.judge,
                    turn_spec.prompt,
                    turn["answer"],
                    summaries,
                    reviewer_guidance=case.reviewer_guidance,
                )
            else:
                turn["judge"] = None
            turns.append(turn)
            if all_messages:
                message_history = all_messages
        return CaseExecution(case_id=case.case_id, status="completed", patient=patient, turns=tuple(turns))
    except Exception as exc:
        # Bounded, generic message only — full exception text can embed
        # patient-identifying detail from some DB/driver errors.
        return CaseExecution(
            case_id=case.case_id,
            status="failed",
            patient=patient,
            turns=(),
            error_type=type(exc).__name__,
            error_message=str(exc)[:500],
        )


async def execute_cases(
    cases: Sequence[NormalizedCase],
    resources: EvaluationResources,
    on_case_complete: Callable[[CaseExecution], None],
) -> list[CaseExecution]:
    """Run every case in order, invoking `on_case_complete` after each one so
    callers can persist incremental progress. A failed case does not stop
    its siblings."""
    results = []
    for case in cases:
        result = await execute_case(case, resources)
        on_case_complete(result)
        results.append(result)
    return results
