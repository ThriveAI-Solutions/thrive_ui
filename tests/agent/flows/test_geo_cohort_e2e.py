"""End-to-end acceptance test: 'give me a list of people in zip 14223 with
high blood pressure' must route through search_codes + search_patients_by_criteria
and must NOT invoke run_sql.

Uses the same FunctionModel scripting pattern as test_cohort_flow.py.
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
    ToolCallStarted,
)


_SQL_FILE = Path(__file__).parents[2] / "agent" / "redshift_synthetic.sql"


def _make_threadsafe_db():
    """Same pattern as tests/agent/flows/test_cohort_flow.py."""
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


def _scripted_llm_geo_cohort():
    """Scripted FunctionModel that mimics LLM routing for the failing question.

    Turn 1: search_codes(vocabulary='icd10', query='hypertension')
    Turn 2: search_patients_by_criteria(diagnosis_codes=['I10'], zip_code='14223')
    Turn 3: final_result(...)
    """
    turn = {"n": 0}

    async def behavior(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        turn["n"] += 1
        if turn["n"] == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="search_codes",
                        args={"vocabulary": "icd10", "query": "hypertension"},
                        tool_call_id="call-1",
                    ),
                ]
            )
        if turn["n"] == 2:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="search_patients_by_criteria",
                        args={"diagnosis_codes": ["I10"], "zip_code": "14223"},
                        tool_call_id="call-2",
                    ),
                ]
            )
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="final_result",
                    args={
                        "text": ("Found 1 patient in zip code 14223 with hypertension (I10)."),
                        "followups": [],
                        "artifacts": [],
                        "clear_selection": False,
                        "cap_reached": False,
                    },
                    tool_call_id="call-3",
                ),
            ]
        )

    return FunctionModel(behavior, model_name="scripted-geo-cohort")


@pytest.mark.asyncio
async def test_zip_plus_dx_question_routes_to_cohort_not_run_sql():
    """The original failing question ('people in zip 14223 with high blood pressure')
    must go through search_codes + search_patients_by_criteria, never run_sql.

    Acceptance criteria:
    - search_patients_by_criteria appears in ToolCallStarted events
    - run_sql does NOT appear in any ToolCallStarted event
    - CohortSampleEvent is emitted with data_availability == 'data_present'
    - The cohort tool returned data (seeded I10 row for patient in zip 14223)
    """
    engine = _make_threadsafe_db()

    # Seed an I10 diagnosis for patient_id=1 (src-john-1962, zip 14223).
    # The synthetic DB already has E11.9 for patient_id=1 and I10 for
    # patient_id=2 (zip 14201), but not I10 for patient_id=1 (zip 14223).
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO metric_federated_data_v "
                "(patient_id, code, code_type, start_date, is_claims_data) "
                "VALUES (1, 'I10', 'ICD-10', '2025-01-01', 0)"
            )
        )

    runner = AgenticRunner(model=_scripted_llm_geo_cohort())

    events = []
    async for ev in runner.stream(
        "give me a list of people in zip code 14223 with high blood pressure",
        deps=_deps(engine),
    ):
        events.append(ev)

    # Collect tool names from ToolCallStarted events
    started_tool_names = [e.tool_name for e in events if isinstance(e, ToolCallStarted)]

    # search_patients_by_criteria must have been called
    assert "search_patients_by_criteria" in started_tool_names, (
        f"Expected search_patients_by_criteria in tool calls; got {started_tool_names}"
    )

    # run_sql must NOT have been called
    assert "run_sql" not in started_tool_names, (
        f"run_sql should not be invoked for geo-cohort question; got {started_tool_names}"
    )

    # CohortSampleEvent must be emitted with data_present
    cohort_events = [e for e in events if isinstance(e, CohortSampleEvent)]
    assert len(cohort_events) == 1, f"Expected exactly 1 CohortSampleEvent; got {len(cohort_events)}"
    payload = cohort_events[0].payload
    assert payload["data_availability"] == "data_present", f"Expected data_present; got {payload['data_availability']}"
    assert len(payload["sample"]) >= 1, f"Expected at least 1 patient in sample; got {payload['sample']}"

    # Cohort sample must come after the ToolCallCompleted for search_patients_by_criteria
    tcc_index = next(
        i
        for i, e in enumerate(events)
        if isinstance(e, ToolCallCompleted) and e.tool_name == "search_patients_by_criteria"
    )
    cse_index = events.index(cohort_events[0])
    assert cse_index > tcc_index, (
        f"CohortSampleEvent (index {cse_index}) must follow "
        f"ToolCallCompleted for search_patients_by_criteria (index {tcc_index})"
    )

    # FinalResponseEvent must be present
    final_events = [e for e in events if isinstance(e, FinalResponseEvent)]
    assert len(final_events) == 1, f"Expected exactly 1 FinalResponseEvent; got {len(final_events)}"
