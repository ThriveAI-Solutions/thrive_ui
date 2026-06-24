"""Unit tests for the Vanna-fallback decision engine (Feature #230 of Epic #228).

Three primitives under test:
  - ``detect_trigger`` — deterministic event classification.
  - ``classify_answer_adequacy`` — 1-shot LLM judge (mocked).
  - ``should_fallback`` — orchestrator that combines them.

LLM calls in the classifier tests are mocked via pydantic-ai's TestModel
(``custom_output_text``) for the happy/sad paths, and via a module-level
patch of ``agent.fallback.Agent`` for the error path. No real provider
calls are made.
"""

from __future__ import annotations

import pytest
from pydantic_ai.models.test import TestModel

from agent.fallback import (
    FallbackClassifierError,
    FallbackDecision,
    SQL_TOOL_NAMES,
    classify_answer_adequacy,
    detect_trigger,
    should_fallback,
)
from agent.state import ToolCallStarted


def _event(tool_name: str) -> ToolCallStarted:
    return ToolCallStarted(tool_name=tool_name, arguments={})


# ----- detect_trigger -----------------------------------------------------


def test_sql_tool_names_constant_contents():
    assert SQL_TOOL_NAMES == frozenset({"run_sql", "get_patient_clinical_data", "search_patients_by_criteria"})


def test_detect_trigger_zero_tools():
    assert detect_trigger([]) == "zero_tools"


def test_detect_trigger_no_sql_tool_called():
    events = [_event("search_knowledge_base"), _event("find_patient")]
    assert detect_trigger(events) == "no_sql_tool"


def test_detect_trigger_sql_tool_called():
    events = [_event("search_knowledge_base"), _event("run_sql")]
    assert detect_trigger(events) == "sql_tool_called"


def test_detect_trigger_get_patient_clinical_data_counts():
    assert detect_trigger([_event("get_patient_clinical_data")]) == "sql_tool_called"


def test_detect_trigger_search_patients_by_criteria_counts():
    assert detect_trigger([_event("search_patients_by_criteria")]) == "sql_tool_called"


# ----- classify_answer_adequacy ------------------------------------------


@pytest.mark.asyncio
async def test_classify_answer_adequacy_yes():
    model = TestModel(custom_output_text="ADEQUATE")
    assert await classify_answer_adequacy("q", "answer", model=model) is True


@pytest.mark.asyncio
async def test_classify_answer_adequacy_no():
    model = TestModel(custom_output_text="INADEQUATE")
    assert await classify_answer_adequacy("q", "answer", model=model) is False


@pytest.mark.asyncio
async def test_classify_answer_adequacy_ambiguous_treated_as_inadequate():
    model = TestModel(custom_output_text="maybe sort of yes")
    assert await classify_answer_adequacy("q", "answer", model=model) is False


@pytest.mark.asyncio
async def test_classify_answer_adequacy_lenient_whitespace_and_case():
    model = TestModel(custom_output_text="  adequate\n")
    assert await classify_answer_adequacy("q", "answer", model=model) is True


@pytest.mark.asyncio
async def test_classify_answer_adequacy_raises_on_llm_error(monkeypatch):
    class _FailingAgent:
        def __init__(self, *_a, **_k):
            pass

        async def run(self, *_a, **_k):
            raise RuntimeError("LLM down")

    monkeypatch.setattr("agent.fallback.Agent", _FailingAgent)
    with pytest.raises(FallbackClassifierError, match="LLM down"):
        await classify_answer_adequacy("q", "answer", model="injected-sentinel")


# ----- should_fallback ----------------------------------------------------


@pytest.mark.asyncio
async def test_should_fallback_feature_disabled_short_circuits(monkeypatch):
    called = {"classifier": False}

    async def _spy(*_a, **_k):
        called["classifier"] = True
        return True

    monkeypatch.setattr("agent.fallback.classify_answer_adequacy", _spy)

    decision = await should_fallback(
        question="q",
        final_text="answer",
        tool_events=[],
        feature_enabled=False,
    )
    assert decision == FallbackDecision(invoke=False, reason="feature_disabled")
    assert called["classifier"] is False


@pytest.mark.asyncio
async def test_should_fallback_sql_tool_called_short_circuits(monkeypatch):
    called = {"classifier": False}

    async def _spy(*_a, **_k):
        called["classifier"] = True
        return True

    monkeypatch.setattr("agent.fallback.classify_answer_adequacy", _spy)

    decision = await should_fallback(
        question="q",
        final_text="answer",
        tool_events=[_event("run_sql")],
        feature_enabled=True,
    )
    assert decision == FallbackDecision(invoke=False, reason="sql_tool_called")
    assert called["classifier"] is False


@pytest.mark.asyncio
async def test_should_fallback_classifier_says_inadequate_returns_invoke_true(monkeypatch):
    async def _classify(*_a, **_k):
        return False

    monkeypatch.setattr("agent.fallback.classify_answer_adequacy", _classify)

    decision = await should_fallback(
        question="how many patients?",
        final_text="I'm not sure what you mean.",
        tool_events=[_event("search_knowledge_base")],
        feature_enabled=True,
    )
    assert decision == FallbackDecision(invoke=True, reason="classifier_says_inadequate")


@pytest.mark.asyncio
async def test_should_fallback_classifier_says_adequate_returns_invoke_false(monkeypatch):
    async def _classify(*_a, **_k):
        return True

    monkeypatch.setattr("agent.fallback.classify_answer_adequacy", _classify)

    decision = await should_fallback(
        question="what does hie_consent mean?",
        final_text="HIE consent is a flag indicating ...",
        tool_events=[],
        feature_enabled=True,
    )
    assert decision == FallbackDecision(invoke=False, reason="classifier_says_adequate")


@pytest.mark.asyncio
async def test_should_fallback_classifier_error_fails_closed(monkeypatch):
    async def _raising(*_a, **_k):
        raise FallbackClassifierError("provider down")

    monkeypatch.setattr("agent.fallback.classify_answer_adequacy", _raising)

    decision = await should_fallback(
        question="q",
        final_text="answer",
        tool_events=[],
        feature_enabled=True,
    )
    assert decision.invoke is False
    assert decision.reason == "classifier_error"
    assert decision.classifier_error == "provider down"
