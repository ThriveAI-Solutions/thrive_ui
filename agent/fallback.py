"""Fallback decision engine — pure logic for Epic #228.

Decides whether an agent turn that finished without retrieving data should
fall back to the legacy Vanna SQL pipeline. Three primitives:

  - ``detect_trigger`` — deterministic check on the turn's tool events.
  - ``classify_answer_adequacy`` — 1-shot LLM judge of the agent's final text.
  - ``should_fallback`` — orchestrates the two.

NO Streamlit access here — this module is imported by the agent runtime
(``agent/runtime.py``) which may be running on the asyncio loop thread.
``feature_enabled`` and ``scrubbed`` flags are passed in by the caller, not
read from secrets.

Fail-closed: when the classifier itself errors, the orchestrator returns
``invoke=False``. The Epic's Acceptance Criteria require "Classifier
failure does not block the user — the original agent reply renders,
fallback is skipped."

Wiring into the runtime lives in Feature #231.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic_ai import Agent

from agent.models import build_model
from agent.state import ToolCallCompleted, ToolCallStarted


SQL_TOOL_NAMES = frozenset(
    {
        "run_sql",
        "get_patient_clinical_data",
        "search_patients_by_criteria",
    }
)


_FallbackReason = Literal[
    "feature_disabled",
    "sql_tool_called",
    "classifier_says_inadequate",
    "classifier_says_adequate",
    "classifier_error",
]


_TriggerKind = Literal["zero_tools", "no_sql_tool", "sql_tool_called"]


class FallbackClassifierError(Exception):
    """Raised by ``classify_answer_adequacy`` when the LLM call fails.

    The orchestrator catches this and returns a fail-closed
    ``FallbackDecision`` so a classifier outage never blocks the user.
    """


@dataclass(frozen=True)
class FallbackDecision:
    invoke: bool
    reason: _FallbackReason
    classifier_error: str | None = None


def detect_trigger(tool_events: list[ToolCallStarted | ToolCallCompleted]) -> _TriggerKind:
    """Classify the turn's tool activity for fallback purposes.

    Returns ``"sql_tool_called"`` if any event targets a data-answer tool
    (see ``SQL_TOOL_NAMES``), ``"zero_tools"`` if the list is empty, else
    ``"no_sql_tool"``. Pure function: no Streamlit, no LLM, deterministic.
    """
    if not tool_events:
        return "zero_tools"
    for event in tool_events:
        if event.tool_name in SQL_TOOL_NAMES:
            return "sql_tool_called"
    return "no_sql_tool"


_CLASSIFIER_PROMPT_TEMPLATE = (
    "You are reviewing an AI assistant's response to a user's question.\n"
    "Question: {question}\n"
    "Response: {final_text}\n"
    "Did the response actually answer the user's data question with concrete "
    "data, or did the assistant decline / give a non-data answer?\n"
    "Reply with exactly one word: ADEQUATE or INADEQUATE."
)


async def classify_answer_adequacy(
    question: str,
    final_text: str,
    *,
    scrubbed: bool = False,
    model: Any | None = None,
) -> bool:
    """Ask the configured LLM whether ``final_text`` adequately answers
    ``question``. Returns ``True`` only when the LLM unambiguously replies
    ``ADEQUATE`` (case- and whitespace-insensitive). Anything else —
    including gibberish or ``INADEQUATE`` — returns ``False``. Conservative
    bias: when in doubt, assume the user's data question was NOT answered.

    Raises ``FallbackClassifierError`` on any LLM exception so the
    orchestrator can fail-closed.

    The ``model`` kwarg exists for test injection. In production, leave it
    ``None`` and the function builds the configured model via
    ``agent.models.build_model()`` — so the user's selected provider /
    model is honored implicitly.

    When ``scrubbed=True`` the prompt should be PHI-scrubbed before the
    LLM call. Real wiring deferred — see TODO below.
    """
    if scrubbed:
        # TODO(#228): wire agent.run_logger scrub helpers when available
        # so the classifier prompt honors [agent_logging].mode = "scrubbed".
        # Until then, scrubbed=True is a signal-only flag and the prompt
        # contains verbatim question + final_text.
        pass

    chosen_model = model if model is not None else build_model()
    agent: Agent[None, str] = Agent(chosen_model, output_type=str)
    prompt = _CLASSIFIER_PROMPT_TEMPLATE.format(question=question, final_text=final_text)
    try:
        result = await agent.run(prompt)
    except Exception as exc:
        raise FallbackClassifierError(str(exc)) from exc

    raw_output = getattr(result, "output", "")
    normalized = (raw_output or "").strip().upper()
    return normalized == "ADEQUATE"


async def should_fallback(
    *,
    question: str,
    final_text: str,
    tool_events: list[ToolCallStarted | ToolCallCompleted],
    feature_enabled: bool,
    scrubbed: bool = False,
) -> FallbackDecision:
    """Decide whether to invoke the Vanna fallback for this turn.

    Decision tree:
      1. ``feature_enabled=False`` → no fallback (short-circuit; no LLM
         call).
      2. The turn already called a data-answer tool → no fallback
         (short-circuit; no LLM call).
      3. Otherwise classify the agent's final text:
         - ``ADEQUATE`` → no fallback.
         - anything else → fallback.
         - classifier error → fail-closed, no fallback.
    """
    if not feature_enabled:
        return FallbackDecision(invoke=False, reason="feature_disabled")

    trigger = detect_trigger(tool_events)
    if trigger == "sql_tool_called":
        return FallbackDecision(invoke=False, reason="sql_tool_called")

    try:
        adequate = await classify_answer_adequacy(question, final_text, scrubbed=scrubbed)
    except FallbackClassifierError as exc:
        return FallbackDecision(
            invoke=False,
            reason="classifier_error",
            classifier_error=str(exc),
        )

    if adequate:
        return FallbackDecision(invoke=False, reason="classifier_says_adequate")
    return FallbackDecision(invoke=True, reason="classifier_says_inadequate")
