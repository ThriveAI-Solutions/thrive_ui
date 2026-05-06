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


class FinalResponseEvent(BaseModel):
    kind: Literal["final_response"] = "final_response"
    response: "AgentResponse"


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


StreamEvent = Annotated[
    Union[
        ToolCallStarted,
        ToolCallCompleted,
        FinalResponseEvent,
        CapReachedEvent,
        PatientChooserEvent,
    ],
    Field(discriminator="kind"),
]


# --- Final response --------------------------------------------------


class Artifact(BaseModel):
    """Variant definitions deferred to Phase 3 (chart, summary, dataframe).

    For Phase 1 we only need the wrapper shape so AgentResponse validates.
    """

    artifact_type: str
    payload: dict


class AgentResponse(BaseModel):
    text: str
    followups: List[str] = []
    artifacts: List[Artifact] = []
    clear_selection: bool = False
    cap_reached: bool = False


FinalResponseEvent.model_rebuild()
