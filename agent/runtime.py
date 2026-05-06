"""Streamlit-side runtime for the agentic message flow.

Replaces the Vanna call chain inside chat_bot_helper.normal_message_flow
when the user has agentic_mode=True.
"""

from __future__ import annotations
import asyncio
import json
import threading
from typing import Any, Coroutine

import streamlit as st

from agent.deps_builder import build_agent_deps
from agent.observability import configure_observability
from agent.runner import AgenticRunner
from agent.state import (
    AgentResponse,
    StreamEvent,
    ToolCallStarted,
    ToolCallCompleted,
    FinalResponseEvent,
    CapReachedEvent,
)
from utils.enums import MessageType, RoleType
from orm.models import Message, SessionLocal


@st.cache_resource
def _runner() -> AgenticRunner:
    configure_observability()
    return AgenticRunner()


def _run_async(coro: Coroutine[Any, Any, Any]) -> Any:
    """Safely execute a coroutine to completion regardless of caller context.

    asyncio.run() raises if a loop is already running (e.g., when called
    from inside another coroutine, or under Streamlit + Tornado on some
    versions). Spawn a worker thread with a fresh loop in that case.
    """
    try:
        asyncio.get_running_loop()
        in_loop = True
    except RuntimeError:
        in_loop = False

    if not in_loop:
        return asyncio.run(coro)

    box: dict[str, Any] = {}

    def _worker() -> None:
        try:
            box["value"] = asyncio.run(coro)
        except BaseException as exc:  # propagate to caller thread
            box["error"] = exc

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join()
    if "error" in box:
        raise box["error"]
    return box.get("value")


def run_agentic_message_flow(my_question: str) -> None:
    """Drop-in replacement for the Vanna branch in normal_message_flow.

    Renders user message, runs the agent with streaming, persists
    each tool call and the final response as orm.Message rows.
    """
    from utils.chat_bot_helper import add_message

    add_message(
        Message(
            RoleType.USER,
            my_question,
            MessageType.TEXT,
        )
    )

    sqlite_session = SessionLocal()
    try:
        deps = build_agent_deps(sqlite_session)
        runner = _runner()

        async def consume() -> None:
            async for event in runner.stream(my_question, deps=deps):
                _render_event(event)

        _run_async(consume())
        sqlite_session.commit()
    finally:
        sqlite_session.close()


def _render_event(event: StreamEvent) -> None:
    from utils.chat_bot_helper import add_message

    if isinstance(event, ToolCallStarted):
        # In-flight tool call card.
        payload = {
            "tool_name": event.tool_name,
            "arguments": event.arguments,
            "result_summary": None,
            "success": None,
            "elapsed_ms": None,
            "error": None,
        }
        add_message(
            Message(
                RoleType.ASSISTANT,
                json.dumps(payload),
                MessageType.TOOL_CALL,
            )
        )
        return

    if isinstance(event, ToolCallCompleted):
        # Completed tool call card.
        payload = {
            "tool_name": event.tool_name,
            "arguments": {},
            "result_summary": event.result_summary,
            "success": event.success,
            "elapsed_ms": event.elapsed_ms,
            "error": event.error,
        }
        add_message(
            Message(
                RoleType.ASSISTANT,
                json.dumps(payload),
                MessageType.TOOL_CALL,
            )
        )
        return

    if isinstance(event, FinalResponseEvent):
        response = event.response
        # If the response carries a PatientSearchResults artifact (Phase 1
        # surfaces this via a tool result), render the chooser.
        for artifact in response.artifacts:
            if artifact.artifact_type == "patient_search_results":
                add_message(
                    Message(
                        RoleType.ASSISTANT,
                        json.dumps(artifact.payload),
                        MessageType.PATIENT_CHOOSER,
                    )
                )
        # Final text
        add_message(
            Message(
                RoleType.ASSISTANT,
                response.text,
                MessageType.TEXT,
            )
        )
        # Honor clear_selection
        if response.clear_selection:
            for k in (
                "selected_patient_source_id",
                "selected_patient_display_name",
                "selected_patient_dob",
                "selection_origin",
                "selected_at",
            ):
                st.session_state.pop(k, None)
        return

    if isinstance(event, CapReachedEvent):
        add_message(
            Message(
                RoleType.ASSISTANT,
                f"Stopped: cap reached ({event.reason}).",
                MessageType.ERROR,
            )
        )
