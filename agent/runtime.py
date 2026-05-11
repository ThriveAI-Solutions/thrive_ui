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
    CapReachedEvent,
    CohortSampleEvent,
    FinalResponseEvent,
    PatientChooserEvent,
    StreamEvent,
    ToolCallCompleted,
    ToolCallStarted,
)
from utils.enums import MessageType, RoleType
from orm.models import Message, SessionLocal


@st.cache_resource
def _runner() -> AgenticRunner:
    configure_observability()
    return AgenticRunner()


_LOOP: asyncio.AbstractEventLoop | None = None
_LOOP_LOCK = threading.Lock()


def _persistent_loop() -> asyncio.AbstractEventLoop:
    """Process-wide event loop reused across requests.

    Pydantic AI's OpenAI / Anthropic providers hold long-lived httpx
    connection pools whose TCP transports are bound to the loop on
    which they were first awaited. asyncio.run() creates and closes a
    fresh loop per call, leaving the cached client with dead transports
    on the second request — visible as
    "RuntimeError: unable to perform operation on <TCPTransport closed>".
    Pinning a single loop keeps those transports alive.
    """
    global _LOOP
    with _LOOP_LOCK:
        if _LOOP is None or _LOOP.is_closed():
            _LOOP = asyncio.new_event_loop()
        return _LOOP


def _run_async(coro: Coroutine[Any, Any, Any]) -> Any:
    """Safely execute a coroutine to completion regardless of caller context.

    Three paths:
    1. No running loop on this thread → run on the persistent loop via
       run_until_complete (the common Streamlit script-thread case).
    2. Already inside a running loop on this thread (e.g. tests with
       pytest-asyncio) → spawn a worker thread with a fresh asyncio.run
       so we do not deadlock the caller's loop.
    """
    try:
        asyncio.get_running_loop()
        in_loop = True
    except RuntimeError:
        in_loop = False

    if not in_loop:
        return _persistent_loop().run_until_complete(coro)

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

    Multi-turn continuity: prior `agent_message_history` (list of
    pydantic-ai ModelMessage) is threaded into the run; the post-run
    transcript is written back. The original question is also stashed
    as `pending_user_question` so the patient-chooser click handler
    can re-trigger this flow with the same prompt.
    """
    from utils.chat_bot_helper import add_message

    # Stash the question for chooser-click resume before any errors can
    # short-circuit us out.
    st.session_state["pending_user_question"] = my_question

    # Outer try ensures my_question is always cleared, even if
    # add_message or SessionLocal raises before we reach the agent
    # call — otherwise the chat_bot.py loop would re-fire the same
    # question on the next rerun and the user would see no response.
    try:
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
            prior_history = st.session_state.get("agent_message_history") or None

            async def consume() -> None:
                async for event in runner.stream(my_question, deps=deps, message_history=prior_history):
                    _render_event(event)

            _run_async(consume())
            sqlite_session.commit()
        finally:
            sqlite_session.close()
    finally:
        # Parity with the Vanna flow at chat_bot_helper.py:1417.
        st.session_state["my_question"] = None


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
            "reliability_note": event.reliability_note,
            "sql_executed": event.sql_executed,
            "result_payload": event.result_payload,
        }
        add_message(
            Message(
                RoleType.ASSISTANT,
                json.dumps(payload),
                MessageType.TOOL_CALL,
            )
        )
        return

    if isinstance(event, PatientChooserEvent):
        # Auto-surfaced by runner.stream() right after find_patient succeeds,
        # independent of any artifact the model may attach to final_result.
        add_message(
            Message(
                RoleType.ASSISTANT,
                json.dumps(event.payload),
                MessageType.PATIENT_CHOOSER,
            )
        )
        return

    if isinstance(event, CohortSampleEvent):
        # Auto-surfaced by runner.stream() right after
        # search_patients_by_criteria succeeds with a non-empty sample.
        # Renders as a DataFrame message regardless of whether the LLM
        # attaches a DataFrameArtifact to the final response.
        import pandas as pd

        sample = event.payload.get("sample", [])
        if sample:
            df = pd.DataFrame(sample)
            add_message(
                Message(
                    RoleType.ASSISTANT,
                    df.to_json(orient="records", date_format="iso"),
                    MessageType.DATAFRAME,
                )
            )
        return

    if isinstance(event, FinalResponseEvent):
        response = event.response
        add_message(
            Message(
                RoleType.ASSISTANT,
                response.text,
                MessageType.TEXT,
            )
        )
        # Translate AgentResponse.artifacts into renderable chat
        # messages (Phase 3 design §3.3). The question comes from
        # session_state where it was stashed by run_agentic_message_flow
        # at line 110 — at this point the agent has been called with it.
        question = st.session_state.get("pending_user_question", "")
        try:
            from utils.chat_bot_helper import render_agent_artifacts

            render_agent_artifacts(response, question=question)
        except Exception:
            # Artifact rendering failure should not block the rest of
            # the FinalResponseEvent handling. Log and continue.
            import traceback

            traceback.print_exc()
        # Honor clear_selection: drop both the slot AND the conversation
        # history, since "start fresh" means no carry-over context.
        if response.clear_selection:
            for k in (
                "selected_patient_source_id",
                "selected_patient_display_name",
                "selected_patient_dob",
                "selection_origin",
                "selected_at",
                "agent_message_history",
                "pending_user_question",
            ):
                st.session_state.pop(k, None)
        else:
            # Persist the post-run transcript so the next turn (whether
            # a follow-up or a chooser-click resume) continues the same
            # conversation rather than starting over.
            if event.all_messages:
                st.session_state["agent_message_history"] = list(event.all_messages)
        return

    if isinstance(event, CapReachedEvent):
        add_message(
            Message(
                RoleType.ASSISTANT,
                f"Stopped: cap reached ({event.reason}).",
                MessageType.ERROR,
            )
        )
