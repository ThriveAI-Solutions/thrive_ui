"""Streaming tests: ensure runner.stream() emits ThinkingDelta /
ThinkingCompleted / AssistantTextDelta events when the model produces
ThinkingPart and TextPart deltas.

Uses pydantic-ai's FunctionModel with a stream_function so we don't
need a live Ollama. The shapes the stream_function yields (str,
DeltaToolCall, DeltaThinkingPart) are the supported test surface.
"""

from __future__ import annotations
import pytest
from unittest.mock import MagicMock

from pydantic_ai.models.function import (
    AgentInfo,
    DeltaThinkingPart,
    DeltaToolCall,
    FunctionModel,
)
from pydantic_ai.messages import ModelMessage

from agent.deps import AgentDeps
from agent.runner import AgenticRunner
from agent.state import (
    AssistantTextDeltaEvent,
    FinalResponseEvent,
    ThinkingCompletedEvent,
    ThinkingDeltaEvent,
    ToolCallStarted,
)


def _final_result_call(call_id: str = "c1", index: int = 1) -> dict[int, DeltaToolCall]:
    """Single-shot delta that emits a complete final_result tool call.

    pydantic-ai builds ToolCallParts by accumulating DeltaToolCall items
    keyed by a stable index. The index must not collide with any prior
    part (e.g. ThinkingPart at index 0) — pass index >= 1 when the turn
    also streams thinking.
    """
    args_json = '{"text": "ok", "followups": [], "artifacts": [], "clear_selection": false, "cap_reached": false}'
    return {
        index: DeltaToolCall(
            name="final_result",
            json_args=args_json,
            tool_call_id=call_id,
        )
    }


@pytest.mark.asyncio
async def test_thinking_only_turn_emits_deltas_and_completed():
    """A turn that streams thinking chunks then a final_result call must
    yield: ThinkingDeltaEvent (>=1), then ThinkingCompletedEvent, then
    the call_tools events the existing stream loop already covers."""

    async def stream(_messages: list[ModelMessage], _info: AgentInfo):
        yield {0: DeltaThinkingPart(content="Looking at ")}
        yield {0: DeltaThinkingPart(content="the question. ")}
        yield {0: DeltaThinkingPart(content="Calling final_result.")}
        yield _final_result_call()

    runner = AgenticRunner(model=FunctionModel(stream_function=stream))
    deps = MagicMock(spec=AgentDeps)
    deps.audit_logger = None
    deps.analytics_db = None
    deps.selected_patient = None
    deps.last_dataframe = None

    events = []
    async for ev in runner.stream("hello", deps=deps):
        events.append(ev)

    deltas = [e for e in events if isinstance(e, ThinkingDeltaEvent)]
    completed = [e for e in events if isinstance(e, ThinkingCompletedEvent)]
    finals = [e for e in events if isinstance(e, FinalResponseEvent)]

    assert len(deltas) == 3
    assert "".join(d.delta for d in deltas) == "Looking at the question. Calling final_result."
    assert all(d.turn_index == 1 for d in deltas)

    assert len(completed) == 1
    assert completed[0].turn_index == 1
    assert completed[0].text == "Looking at the question. Calling final_result."
    assert completed[0].elapsed_ms >= 0

    # ThinkingCompleted must arrive before the FinalResponseEvent so the
    # renderer can swap its placeholder for a persisted row before the
    # final text lands.
    assert events.index(completed[0]) < events.index(finals[0])


@pytest.mark.asyncio
async def test_text_deltas_emitted_for_plain_text_parts():
    """If the model yields raw str deltas (TextPart), they should surface
    as AssistantTextDeltaEvent. No ThinkingCompletedEvent because no
    ThinkingPart was streamed."""

    async def stream(_messages: list[ModelMessage], _info: AgentInfo):
        yield "Picking "
        yield "a tool…"
        yield _final_result_call(index=1)

    runner = AgenticRunner(model=FunctionModel(stream_function=stream))
    deps = MagicMock(spec=AgentDeps)
    deps.audit_logger = None
    deps.analytics_db = None
    deps.selected_patient = None
    deps.last_dataframe = None

    events = []
    async for ev in runner.stream("hello", deps=deps):
        events.append(ev)

    text_deltas = [e for e in events if isinstance(e, AssistantTextDeltaEvent)]
    assert [d.delta for d in text_deltas] == ["Picking ", "a tool…"]
    assert all(d.turn_index == 1 for d in text_deltas)
    assert not [e for e in events if isinstance(e, ThinkingCompletedEvent)]


@pytest.mark.asyncio
async def test_no_thinking_no_text_emits_nothing_extra():
    """A turn that only emits a tool call (no thinking, no text) must not
    produce any thinking/text events. Smoke-checks we don't accidentally
    emit empty completed events."""

    async def stream(_messages: list[ModelMessage], _info: AgentInfo):
        yield _final_result_call()

    runner = AgenticRunner(model=FunctionModel(stream_function=stream))
    deps = MagicMock(spec=AgentDeps)
    deps.audit_logger = None
    deps.analytics_db = None
    deps.selected_patient = None
    deps.last_dataframe = None

    events = []
    async for ev in runner.stream("hello", deps=deps):
        events.append(ev)

    assert not [e for e in events if isinstance(e, ThinkingDeltaEvent)]
    assert not [e for e in events if isinstance(e, ThinkingCompletedEvent)]
    assert not [e for e in events if isinstance(e, AssistantTextDeltaEvent)]
    assert any(isinstance(e, FinalResponseEvent) for e in events)


@pytest.mark.asyncio
async def test_turn_index_increments_across_model_requests():
    """Two model_request_node hits in one run must produce thinking events
    with distinct turn_index values, so the renderer scopes placeholders
    correctly."""

    # Using ToolCallStarted to confirm the tool ran between turns,
    # producing a second model request after the tool result lands.
    call_count = {"n": 0}

    def behavior_args() -> dict[int, DeltaToolCall]:
        return {
            1: DeltaToolCall(
                name="search_codes",
                json_args='{"query": "diabetes"}',
                tool_call_id=f"sc-{call_count['n']}",
            )
        }

    async def stream(_messages: list[ModelMessage], _info: AgentInfo):
        call_count["n"] += 1
        if call_count["n"] == 1:
            yield {0: DeltaThinkingPart(content="First turn thinking.")}
            yield behavior_args()
        else:
            yield {0: DeltaThinkingPart(content="Second turn thinking.")}
            yield _final_result_call(index=1)

    # Stub search_codes so the tool call resolves without hitting RAG.
    from agent.tools import search_codes as sc_mod

    async def fake_search_codes(_ctx, query: str, top_k: int = 12):  # noqa: ARG001
        from agent.tools.search_codes import CodeSearchResult

        return CodeSearchResult(matches=[], total_unique=0)

    original = sc_mod.search_codes
    sc_mod.search_codes = fake_search_codes
    try:
        runner = AgenticRunner(model=FunctionModel(stream_function=stream))
        deps = MagicMock(spec=AgentDeps)
        deps.audit_logger = None
        deps.analytics_db = None
        deps.selected_patient = None
        deps.last_dataframe = None

        events = []
        async for ev in runner.stream("anything", deps=deps):
            events.append(ev)
    finally:
        sc_mod.search_codes = original

    completed = [e for e in events if isinstance(e, ThinkingCompletedEvent)]
    assert {c.turn_index for c in completed} == {1, 2}
    assert any(isinstance(e, ToolCallStarted) and e.tool_name == "search_codes" for e in events)
