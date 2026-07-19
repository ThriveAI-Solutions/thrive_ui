"""Model-backed summarizer for Patient 360 (#246).

Wraps thrive's configured LLM (``agent.models.build_model``) as the
``summarizer(system_prompt, table_markdown) -> str`` callable the pipeline
expects. Each call is an independent single-shot completion: the domain (or
synthesis) prompt is the system prompt, the rendered table is the user message.
No tools, no agent loop — deterministic, fresh context per section.
"""

from __future__ import annotations

from typing import Any, Optional

from agent.patient360.pipeline import Summarizer


def build_patient360_summarizer(model: Optional[Any] = None) -> Summarizer:
    """Return a summarizer backed by a pydantic-ai model.

    ``model`` defaults to ``agent.models.build_model()`` (the active provider
    from secrets); pass one to override (e.g. tests or a cheaper 360 model).
    Built lazily so importing this module doesn't require model config.
    """
    from pydantic_ai import Agent

    from agent.models import build_model

    resolved = model if model is not None else build_model()

    def summarize(system_prompt: str, table_markdown: str) -> str:
        agent: Agent[None, str] = Agent(resolved, system_prompt=system_prompt, output_type=str)
        return agent.run_sync(table_markdown).output

    return summarize
