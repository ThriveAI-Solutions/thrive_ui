"""Phase 1 runner tests using Pydantic AI's TestModel for deterministic
LLM behavior. Per spec §11.2."""

import pytest
from pydantic_ai.models.test import TestModel
from agent.runner import AgenticRunner
from agent.deps import AgentDeps
from unittest.mock import MagicMock


@pytest.mark.asyncio
async def test_runner_constructs_with_test_model():
    runner = AgenticRunner(model=TestModel())
    assert runner._agent is not None


@pytest.mark.asyncio
async def test_runner_run_returns_agent_response():
    runner = AgenticRunner(model=TestModel(call_tools=[]))
    deps = MagicMock(spec=AgentDeps)
    response = await runner.run("hello", deps=deps)
    assert hasattr(response, "text")


def test_runner_registers_phase2_tools():
    """list_patient_documents and search_codes must be registered alongside Phase 1 tools."""
    from agent.runner import AgenticRunner
    from pydantic_ai import Agent
    from pydantic_ai.models.test import TestModel
    from agent.deps import AgentDeps
    from agent.state import AgentResponse
    from agent.system_prompt import SYSTEM_PROMPT

    runner = AgenticRunner.__new__(AgenticRunner)  # bypass __init__ to avoid model wiring
    runner._agent = Agent(
        model=TestModel(),
        deps_type=AgentDeps,
        output_type=AgentResponse,
        system_prompt=SYSTEM_PROMPT,
    )
    runner._register_tools()
    tool_names = set(runner._agent._function_toolset.tools.keys())
    assert "find_patient" in tool_names
    assert "get_patient_clinical_data" in tool_names
    assert "search_knowledge_base" in tool_names
    assert "list_patient_documents" in tool_names
    assert "search_codes" in tool_names


@pytest.mark.asyncio
async def test_runner_injects_selection_into_instructions():
    """When deps.selected_patient is set, the model must receive the
    selection_instructions text in the prompt — that's how it knows to
    skip find_patient. We capture what the model actually sees by using
    a FunctionModel that records its incoming messages.
    """
    from datetime import date, datetime
    from pydantic_ai.messages import (
        ModelMessage,
        ModelRequest,
        ModelResponse,
        TextPart,
        ToolCallPart,
    )
    from pydantic_ai.models.function import AgentInfo, FunctionModel
    from unittest.mock import MagicMock
    from agent.deps import AgentDeps, SelectedPatient

    captured: dict = {"prompt_text": ""}

    async def behavior(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        for m in messages:
            if isinstance(m, ModelRequest):
                for part in m.parts:
                    text = getattr(part, "content", None)
                    if isinstance(text, str):
                        captured["prompt_text"] += "\n" + text
                    instructions = getattr(m, "instructions", None)
                    if isinstance(instructions, str):
                        captured["prompt_text"] += "\n" + instructions
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="final_result",
                    args={
                        "text": "ok",
                        "followups": [],
                        "artifacts": [],
                        "clear_selection": False,
                        "cap_reached": False,
                    },
                    tool_call_id="c1",
                ),
            ]
        )

    runner = AgenticRunner(model=FunctionModel(behavior, model_name="capture"))
    deps = MagicMock(spec=AgentDeps)
    deps.selected_patient = SelectedPatient(
        source_id="src-john-1962",
        display_name="John Smith",
        dob=date(1962, 5, 1),
        selected_at=datetime(2026, 5, 6),
        selection_origin="user_click",
    )
    deps.run_logger = None

    await runner.run("anything", deps=deps)
    assert "src-john-1962" in captured["prompt_text"]
    assert "find_patient" in captured["prompt_text"].lower()


def test_runner_registers_run_sql():
    from agent.runner import AgenticRunner
    from pydantic_ai.models.test import TestModel

    runner = AgenticRunner(model=TestModel())
    tool_names = {t.name for t in runner._agent._function_toolset.tools.values()}
    assert "run_sql" in tool_names


def test_runner_registers_make_chart():
    from agent.runner import AgenticRunner
    from pydantic_ai.models.test import TestModel

    runner = AgenticRunner(model=TestModel())
    tool_names = {t.name for t in runner._agent._function_toolset.tools.values()}
    assert "make_chart" in tool_names


def test_runner_registers_summarize_results():
    from agent.runner import AgenticRunner
    from pydantic_ai.models.test import TestModel

    runner = AgenticRunner(model=TestModel())
    tool_names = {t.name for t in runner._agent._function_toolset.tools.values()}
    assert "summarize_results" in tool_names


def test_runner_registers_search_patients_by_criteria():
    from agent.runner import AgenticRunner
    from pydantic_ai.models.test import TestModel

    runner = AgenticRunner(model=TestModel())
    tool_names = {t.name for t in runner._agent._function_toolset.tools.values()}
    assert "search_patients_by_criteria" in tool_names


# --- tool-result outcome classification (2026-07-06 incident) --------------
#
# FunctionToolResultEvent fires for BOTH ToolReturnPart (tool succeeded) and
# RetryPromptPart (tool raised — pydantic ValidationError or ModelRetry). The
# runner used to log every result as success=True / "empty_result", which
# disguised the drug_supply_days validation blowup as an empty medication
# list for a whole evening of forensics.

from pydantic_ai.messages import RetryPromptPart, ToolReturnPart  # noqa: E402
from agent.runner import _tool_result_outcome  # noqa: E402


def test_tool_return_part_is_success():
    part = ToolReturnPart(tool_name="t", content={"items": []}, tool_call_id="c1")
    success, error = _tool_result_outcome(part)
    assert success is True
    assert error is None


def test_retry_prompt_string_is_failure_with_message():
    part = RetryPromptPart(content="No patient is currently selected.", tool_name="t", tool_call_id="c1")
    success, error = _tool_result_outcome(part)
    assert success is False
    assert "No patient is currently selected" in error


def test_retry_prompt_validation_errors_keep_loc_and_type_drop_input():
    """Pydantic ErrorDetails carry the offending input value, which can be
    PHI — the logged error must name the field and error type only."""
    details = [
        {
            "type": "int_parsing",
            "loc": ("items", 3, "drug_supply_days"),
            "msg": "Input should be a valid integer, unable to parse string as an integer",
            "input": "PHI-LADEN-VALUE",
        }
    ]
    part = RetryPromptPart(content=details, tool_name="t", tool_call_id="c1")
    success, error = _tool_result_outcome(part)
    assert success is False
    assert "items.3.drug_supply_days" in error
    assert "int_parsing" in error
    assert "PHI-LADEN-VALUE" not in error
