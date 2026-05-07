"""Per-run dynamic instructions for the agent.

Pydantic-AI's `@agent.instructions` decorator runs these on every agent
invocation, with the live RunContext. Unlike the static system prompt,
returned text does not get persisted into message history — exactly the
right lifecycle for transient slot state like the currently-selected
patient.
"""

from __future__ import annotations

from pydantic_ai import RunContext

from agent.deps import AgentDeps


def selection_instructions(ctx: RunContext[AgentDeps]) -> str:
    """Tell the agent which patient is currently selected, if any.

    Empty string when no slot is filled — keeps default behavior unchanged
    (system prompt's find_patient-first rule applies). When a patient IS
    selected, instruct the agent to skip find_patient and use the
    patient-specific tools directly.
    """
    sp = getattr(ctx.deps, "selected_patient", None)
    if sp is None:
        return ""

    dob_str = sp.dob.isoformat() if sp.dob else "unknown"
    return (
        f"A patient is currently selected: {sp.display_name}, "
        f"source_id '{sp.source_id}', DOB {dob_str}. "
        f"Do NOT call find_patient — the slot is already filled. "
        f"Use the patient-specific tools (get_patient_clinical_data, "
        f"list_patient_documents) which read the slot automatically."
    )
