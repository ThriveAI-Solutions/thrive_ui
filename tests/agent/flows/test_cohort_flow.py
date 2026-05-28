"""Phase 4 §3.7 — the runtime auto-surfaces a DataFrame message after
search_patients_by_criteria succeeds with a non-empty sample.

Mirrors the find_patient PatientChooserEvent auto-surface pattern.
"""

from __future__ import annotations
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart

from agent.runner import AgenticRunner
from agent.deps import AgentDeps
from agent.db.analytics_adapter import AnalyticsDbAdapter
from agent.state import (
    CohortSampleEvent,
    FinalResponseEvent,
    ToolCallCompleted,
)


_SQL_FILE = Path(__file__).parents[2] / "agent" / "redshift_synthetic.sql"


def _make_threadsafe_db():
    """Same pattern as tests/agent/flows/test_disambiguation_flow.py."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    sql = _SQL_FILE.read_text()
    with engine.begin() as conn:
        for stmt in sql.split(";"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(text(stmt))
    return engine


def _deps(engine) -> AgentDeps:
    return AgentDeps(
        user_id=1,
        user_role=MagicMock(value=1),
        session_id="s1",
        selected_patient=None,
        last_dataframe=None,
        last_sql=None,
        last_query_meta=None,
        analytics_db=AnalyticsDbAdapter(engine=engine, dialect="sqlite"),
        rag=MagicMock(),
        sqlite_session=None,
        run_logger=MagicMock(),
    )


def _scripted_llm_cohort(criteria_args: dict):
    """Scripted FunctionModel that calls search_patients_by_criteria with
    the provided args, then returns a final_result on the second turn.
    """
    turn = {"n": 0}

    async def behavior(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        turn["n"] += 1
        if turn["n"] == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="search_patients_by_criteria",
                        args=criteria_args,
                        tool_call_id="call-1",
                    ),
                ]
            )
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="final_result",
                    args={
                        "text": "Cohort retrieved.",
                        "followups": [],
                        "artifacts": [],
                        "clear_selection": False,
                        "cap_reached": False,
                    },
                    tool_call_id="call-2",
                ),
            ]
        )

    return FunctionModel(behavior, model_name="scripted-cohort")


@pytest.mark.asyncio
async def test_cohort_tool_completion_emits_cohort_sample_event():
    """End-to-end via the streaming runner. The expected order is:
    ToolCallStarted(search_patients_by_criteria)
    ToolCallCompleted(search_patients_by_criteria)
    CohortSampleEvent(payload contains sample)
    FinalResponseEvent
    """
    engine = _make_threadsafe_db()
    runner = AgenticRunner(model=_scripted_llm_cohort({"diagnosis_codes": ["E11.9"], "sample_size": 5}))

    events = []
    async for ev in runner.stream("how many diabetics?", deps=_deps(engine)):
        events.append(ev)

    cohort_events = [e for e in events if isinstance(e, CohortSampleEvent)]
    assert len(cohort_events) == 1, f"expected 1 CohortSampleEvent, got {len(cohort_events)}"
    payload = cohort_events[0].payload
    assert payload["data_availability"] == "data_present"
    assert len(payload["sample"]) >= 1
    # Must come AFTER ToolCallCompleted for the cohort tool
    tcc_index = next(
        i
        for i, e in enumerate(events)
        if isinstance(e, ToolCallCompleted) and e.tool_name == "search_patients_by_criteria"
    )
    cse_index = events.index(cohort_events[0])
    assert cse_index > tcc_index


@pytest.mark.asyncio
async def test_cohort_breakdown_buckets_emit_cohort_sample_event():
    """End-to-end via the streaming runner, exercising the BREAKDOWN path.

    A single-dimension breakdown returns an empty patient sample but a
    non-empty buckets list. This drives the runner.py gate (runner.py
    ~line 539) that also fires CohortSampleEvent when buckets are present,
    not just when a sample is. Mirrors the SAMPLE-path test above exactly,
    only differing in the scripted criteria args (adds breakdown=["gender"]).
    """
    engine = _make_threadsafe_db()
    runner = AgenticRunner(
        model=_scripted_llm_cohort({"diagnosis_codes": ["E11.9"], "sample_size": 0, "breakdown": ["gender"]})
    )

    events = []
    async for ev in runner.stream("break down diabetics by gender", deps=_deps(engine)):
        events.append(ev)

    cohort_events = [e for e in events if isinstance(e, CohortSampleEvent)]
    assert len(cohort_events) == 1, f"expected 1 CohortSampleEvent, got {len(cohort_events)}"
    payload = cohort_events[0].payload
    assert payload["data_availability"] == "data_present"
    assert payload["breakdown_status"] == "single_dimension"
    assert payload["non_additive"] is False
    # The buckets branch of the gate is what we're covering: buckets present,
    # sample empty.
    assert len(payload["buckets"]) >= 1
    assert payload["sample"] == []
    # Must come AFTER ToolCallCompleted for the cohort tool.
    tcc_index = next(
        i
        for i, e in enumerate(events)
        if isinstance(e, ToolCallCompleted) and e.tool_name == "search_patients_by_criteria"
    )
    cse_index = events.index(cohort_events[0])
    assert cse_index > tcc_index


@pytest.mark.asyncio
async def test_cohort_tool_no_event_when_sample_empty():
    """sample_size=0 → no CohortSampleEvent emitted (count-only mode)."""
    engine = _make_threadsafe_db()
    runner = AgenticRunner(model=_scripted_llm_cohort({"diagnosis_codes": ["E11.9"], "sample_size": 0}))

    events = []
    async for ev in runner.stream("count diabetics", deps=_deps(engine)):
        events.append(ev)

    cohort_events = [e for e in events if isinstance(e, CohortSampleEvent)]
    assert len(cohort_events) == 0
