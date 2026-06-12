"""Collect one streamed agent turn into a plain dict for the results JSON.

Mirrors the event handling in scripts/agent_replay.py but returns data
instead of printing. Works with any object exposing
`stream(prompt, deps=..., message_history=...) -> AsyncIterator[StreamEvent]`.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from agent.state import (
    CapReachedEvent,
    FinalResponseEvent,
    ThinkingCompletedEvent,
    ToolCallCompleted,
    ToolCallStarted,
)
from evals.latency import attribute_latency


async def run_turn(
    runner,
    deps,
    prompt: str,
    message_history: Optional[list[Any]] = None,
) -> tuple[dict, list[Any]]:
    tool_calls: list[dict] = []
    thinking: list[str] = []
    answer = ""
    usage: Optional[dict] = None
    cap_reason: Optional[str] = None
    all_messages: list[Any] = []

    started = time.perf_counter()
    async for evt in runner.stream(prompt, deps=deps, message_history=message_history):
        if isinstance(evt, ToolCallStarted):
            tool_calls.append({"tool_name": evt.tool_name, "arguments": evt.arguments, "completed": False})
        elif isinstance(evt, ToolCallCompleted):
            target = None
            for tc in reversed(tool_calls):
                if tc["tool_name"] == evt.tool_name and not tc["completed"]:
                    target = tc
                    break
            if target is None:
                target = {"tool_name": evt.tool_name, "arguments": {}}
                tool_calls.append(target)
            target.update(
                completed=True,
                result_summary=evt.result_summary,
                success=evt.success,
                elapsed_ms=evt.elapsed_ms,
                error=evt.error,
                sql_executed=list(evt.sql_executed or []),
            )
        elif isinstance(evt, ThinkingCompletedEvent):
            thinking.append(evt.text)
        elif isinstance(evt, CapReachedEvent):
            cap_reason = evt.reason
        elif isinstance(evt, FinalResponseEvent):
            answer = evt.response.text
            usage = evt.usage
            all_messages = list(evt.all_messages or [])
    total_ms = int((time.perf_counter() - started) * 1000)

    turn = {
        "prompt": prompt,
        "answer": answer,
        "thinking": thinking,
        "tool_calls": tool_calls,
        "cap_reached": cap_reason,
        "usage": usage,
        "total_elapsed_ms": total_ms,
        "latency": attribute_latency(total_ms, tool_calls),
    }
    return turn, all_messages
