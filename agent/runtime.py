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
    AssistantTextCompletedEvent,
    AssistantTextDeltaEvent,
    CapReachedEvent,
    CohortSampleEvent,
    FinalResponseEvent,
    PatientChooserEvent,
    StreamEvent,
    ThinkingCompletedEvent,
    ThinkingDeltaEvent,
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

    Runs the agent with streaming and persists each tool call and the
    final response as orm.Message rows. The USER row for `my_question`
    is already in session_state.messages — set_question() at
    chat_bot_helper.py:535 wrote it before the rerun landed us here.

    Multi-turn continuity: prior `agent_message_history` (list of
    pydantic-ai ModelMessage) is threaded into the run; the post-run
    transcript is written back. The original question is also stashed
    as `pending_user_question` so the patient-chooser click handler
    can re-trigger this flow with the same prompt.
    """
    # Stash the question for chooser-click resume before any errors can
    # short-circuit us out.
    st.session_state["pending_user_question"] = my_question

    # Outer try ensures my_question is always cleared even if the agent
    # call raises — otherwise the chat_bot.py loop would re-fire the
    # same question on the next rerun and the user would see no response.
    try:
        sqlite_session = SessionLocal()
        try:
            deps = build_agent_deps(sqlite_session)
            runner = _runner()
            prior_history = st.session_state.get("agent_message_history") or None

            async def consume() -> None:
                # Per-flow state for live placeholders. Two buckets keyed by
                # turn_index so thinking and plain assistant text get distinct
                # chat_message containers (different headers, different
                # lifecycles). The finally block drains anything still open
                # if the generator dies mid-stream — otherwise a partial
                # "🤔 Thinking..." panel would stay frozen on screen.
                renderer_state: dict[str, Any] = {"thinking": {}, "text": {}}
                try:
                    async for event in runner.stream(my_question, deps=deps, message_history=prior_history):
                        _render_event(event, renderer_state)
                finally:
                    _drain_placeholders(renderer_state)

            _run_async(consume())
            sqlite_session.commit()
        finally:
            sqlite_session.close()
    finally:
        # Parity with the Vanna flow at chat_bot_helper.py:1417.
        st.session_state["my_question"] = None


def _drain_placeholders(state: dict[str, Any]) -> None:
    """Clear any open transient placeholders. Called from a finally block in
    consume() so a mid-stream death (CapReachedEvent, exception, network
    drop) doesn't leave a partial "🤔 Thinking..." panel frozen on screen.
    Idempotent — safe to call multiple times."""
    for bucket in ("thinking", "text"):
        slots = state.get(bucket) or {}
        for slot in slots.values():
            try:
                slot["outer"].empty()
            except Exception:
                pass
        slots.clear()


def _open_slot(state: dict[str, Any], bucket: str, turn_index: int) -> dict[str, Any]:
    """Return the existing live-update slot for (bucket, turn_index), or
    create one anchored to a fresh outer st.empty().

    The chat_message bubble is rendered INSIDE the outer placeholder on
    each delta via `outer.container()`, so calling `outer.empty()` later
    removes both the bubble's avatar and its content. The prior pattern
    of placing st.empty() inside a chat_message left orphaned empty
    avatar bubbles after the placeholder was cleared.
    """
    slot = state[bucket].get(turn_index)
    if slot is None:
        outer = st.empty()
        slot = {"outer": outer, "buf": ""}
        state[bucket][turn_index] = slot
    return slot


def _write_slot(slot: dict[str, Any], header: str | None = None) -> None:
    """Re-render the slot's buffer into the outer placeholder as a fresh
    chat_message + markdown. Called on every delta. `header` (e.g.
    "🤔 **Thinking...**") is prepended when present."""
    body = (f"{header}\n\n{slot['buf']}") if header else slot["buf"]
    with slot["outer"].container():
        with st.chat_message(RoleType.ASSISTANT.value):
            st.markdown(body)


def _render_event(event: StreamEvent, state: dict[str, Any] | None = None) -> None:
    from utils.chat_bot_helper import add_message

    state = state if state is not None else {"thinking": {}, "text": {}}
    state.setdefault("thinking", {})
    state.setdefault("text", {})

    if isinstance(event, ThinkingDeltaEvent):
        slot = _open_slot(state, "thinking", event.turn_index)
        slot["buf"] += event.delta
        _write_slot(slot, header="🤔 **Thinking...**")
        return

    if isinstance(event, AssistantTextDeltaEvent):
        # Plain assistant narration between tool calls. Lives in its own
        # bucket with no "Thinking..." header so the user can tell the two
        # apart at a glance. Persisted on the matching AssistantTextCompletedEvent.
        slot = _open_slot(state, "text", event.turn_index)
        slot["buf"] += event.delta
        _write_slot(slot)
        return

    if isinstance(event, AssistantTextCompletedEvent):
        slot = state["text"].pop(event.turn_index, None)
        if slot is not None:
            slot["outer"].empty()
        if event.text.strip():
            add_message(
                Message(
                    RoleType.ASSISTANT,
                    event.text,
                    MessageType.TEXT,
                )
            )
            # Remember the last persisted narration so FinalResponseEvent
            # can skip its own response.text row if the model echoed the
            # same content via TextPart on the final turn.
            state["last_persisted_text"] = event.text
        return

    if isinstance(event, ThinkingCompletedEvent):
        slot = state["thinking"].pop(event.turn_index, None)
        if slot is not None:
            # Drop the transient bubble; the persisted MessageType.THINKING
            # row that follows replaces it visually.
            slot["outer"].empty()
        if event.text.strip():
            add_message(
                Message(
                    RoleType.ASSISTANT,
                    event.text,
                    MessageType.THINKING,
                    elapsed_time=event.elapsed_ms / 1000.0,
                )
            )
        return

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
        # Drain any open transient placeholders before the final TEXT row
        # lands, otherwise a plain-text live preview would briefly double up
        # with the persisted answer.
        _drain_placeholders(state)
        response = event.response
        # Dedupe: if the model already streamed the same final text as a
        # TextPart on the last turn, AssistantTextCompletedEvent already
        # persisted it. Skip the redundant row so the user doesn't see the
        # answer twice. Strip-equality is enough — both come from the same
        # model output and won't drift in whitespace meaningfully.
        last_streamed = state.get("last_persisted_text", "")
        if response.text.strip() and response.text.strip() != last_streamed.strip():
            add_message(
                Message(
                    RoleType.ASSISTANT,
                    response.text,
                    MessageType.TEXT,
                )
            )
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
        # Cap fires mid-stream; clear any frozen "Thinking..." panels first.
        _drain_placeholders(state)
        add_message(
            Message(
                RoleType.ASSISTANT,
                f"Stopped: cap reached ({event.reason}).",
                MessageType.ERROR,
            )
        )
