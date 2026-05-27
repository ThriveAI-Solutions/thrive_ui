"""Pydantic shapes for agent run streaming and final response.

StreamEvent variants flow from AgenticRunner.stream() to the chat
renderer, which converts them into Streamlit message updates.

AgentResponse is the structured final output of every agent.run().
"""

from __future__ import annotations
from typing import Annotated, Any, List, Literal, Optional, Union
from pydantic import BaseModel, Field


# --- Stream events ---------------------------------------------------


class ToolCallStarted(BaseModel):
    kind: Literal["tool_call_started"] = "tool_call_started"
    tool_name: str
    arguments: dict


class ToolCallCompleted(BaseModel):
    kind: Literal["tool_call_completed"] = "tool_call_completed"
    tool_name: str
    result_summary: str
    success: bool
    elapsed_ms: int
    error: Optional[str] = None
    reliability_note: Optional[str] = None
    # Observability for the chat UI: SQL the tool ran (popped from
    # adapter.pop_sql_log()) and a serialized snapshot of the tool's
    # return value. Both surfaced in a collapsed expander on every
    # tool-call card (the role gate was removed in Phase 3 §3.7 — see
    # memory/feedback_default_permissive_roles.md). Empty list / None
    # when the tool ran no SQL or the result wasn't dict-serializable.
    sql_executed: List[dict] = Field(default_factory=list)
    result_payload: Optional[dict] = None


class FinalResponseEvent(BaseModel):
    kind: Literal["final_response"] = "final_response"
    response: "AgentResponse"
    # Full transcript of the run (system+user+model messages, tool calls,
    # tool results). Carried so the runtime can persist it back to
    # session_state for multi-turn message_history continuity.
    # Stored as Any to avoid importing pydantic-ai message types into
    # this module's public surface; the runtime treats it as an opaque
    # list of pydantic-ai ModelMessage instances.
    all_messages: list[Any] = Field(default_factory=list)
    # Token usage for this run, from pydantic-ai run.usage(). None when the
    # provider doesn't report usage. Keys: input_tokens, output_tokens, total_tokens.
    usage: Optional[dict] = None

    model_config = {"arbitrary_types_allowed": True}


class CapReachedEvent(BaseModel):
    kind: Literal["cap_reached"] = "cap_reached"
    reason: Literal["wall_clock", "tool_count"]


class PatientChooserEvent(BaseModel):
    """Emitted by the runner immediately after find_patient returns matches,
    so the UI can render a click-to-select chooser without waiting for the
    model to call final_result. Independent of any artifact the model may
    or may not attach.
    """

    kind: Literal["patient_chooser"] = "patient_chooser"
    payload: dict  # PatientSearchResults shape: {matches, total_unique, truncated}


class CohortSampleEvent(BaseModel):
    """Auto-surfaced after a successful search_patients_by_criteria
    call so the UI renders the cohort sample as a DataFrame regardless
    of whether the LLM attaches an artifact to the final response.

    Phase 4 design §3.7.
    """

    kind: Literal["cohort_sample"] = "cohort_sample"
    payload: dict  # CohortResult shape (total_count, sample, data_availability, reliability_note)


class ThinkingDeltaEvent(BaseModel):
    """One chunk of reasoning text from the model's current turn.

    Emitted while a ModelRequest node is streaming a ThinkingPart.
    Renderer accumulates these into a transient placeholder so the
    user can watch the model reason instead of staring at a frozen UI.
    """

    kind: Literal["thinking_delta"] = "thinking_delta"
    delta: str
    turn_index: int  # which model_request_node this delta belongs to


class ThinkingCompletedEvent(BaseModel):
    """End of the current turn's thinking. Carries the full accumulated
    text so the renderer can persist it as a MessageType.THINKING row
    that survives Streamlit reruns.
    """

    kind: Literal["thinking_completed"] = "thinking_completed"
    text: str
    elapsed_ms: int
    turn_index: int


class AssistantTextDeltaEvent(BaseModel):
    """One chunk of plain assistant text from the current turn.

    Rare with output_type=AgentResponse since the model is routed through
    the synthetic final_result tool, but the model can still emit narration
    between tool calls. Rendered live in a transient placeholder; the
    AssistantTextCompletedEvent at end of turn carries the full accumulated
    text so the renderer can persist a row that survives reruns.
    """

    kind: Literal["assistant_text_delta"] = "assistant_text_delta"
    delta: str
    turn_index: int


class AssistantTextCompletedEvent(BaseModel):
    """End of the current turn's plain assistant text. Carries the full
    accumulated text so the renderer can persist it as a MessageType.TEXT
    row that survives Streamlit reruns. Symmetric with ThinkingCompletedEvent.
    """

    kind: Literal["assistant_text_completed"] = "assistant_text_completed"
    text: str
    turn_index: int


StreamEvent = Annotated[
    Union[
        ToolCallStarted,
        ToolCallCompleted,
        FinalResponseEvent,
        CapReachedEvent,
        PatientChooserEvent,
        CohortSampleEvent,
        ThinkingDeltaEvent,
        ThinkingCompletedEvent,
        AssistantTextDeltaEvent,
        AssistantTextCompletedEvent,
    ],
    Field(discriminator="kind"),
]


# --- Final response --------------------------------------------------


class AgentResponse(BaseModel):
    """LLM-facing structured output.

    Deliberately flat — no nested unions or $defs — because local models
    (qwen3.6 on Ollama in particular) silently fail to call the synthetic
    `final_result` tool when its JSON schema gets complex. The artifact
    types still live in agent/artifacts.py for use by individual tools
    (make_chart returns a ChartArtifact, summarize_results returns a
    SummaryArtifact), but those flow through the tool-call card path,
    not via this response object. The patient chooser and cohort sample
    DataFrame are auto-surfaced by AgenticRunner.stream() events.
    """

    text: str
    followups: List[str] = []
    clear_selection: bool = False
    cap_reached: bool = False


FinalResponseEvent.model_rebuild()
