"""summarize_results tool.

Phase 3 design §3.6 — operates on ctx.deps.last_dataframe; wraps the
pure summary shim; returns a SummaryArtifact.
"""

from __future__ import annotations
from typing import Optional

from pydantic import BaseModel, Field
from pydantic_ai import RunContext
from pydantic_ai.exceptions import ModelRetry

from agent.artifacts import SummaryArtifact
from agent.deps import AgentDeps
from utils.summary_shim import generate_summary_for_df


class SummarizeResultsInput(BaseModel):
    question: str = Field(
        ...,
        description=(
            "What the user asked, or what the summary should describe. "
            "The summarizer reads the dataframe and produces prose answering "
            "this question. Examples: 'describe the labs', 'what changed'."
        ),
    )
    focus: Optional[str] = Field(
        None,
        description=(
            "Optional narrowing hint — e.g. 'trend over time', 'outliers', "
            "'most recent value'. Concatenated onto question for the backend."
        ),
    )


def summarize_results(ctx: RunContext[AgentDeps], input: SummarizeResultsInput) -> SummaryArtifact:
    """Produce a prose summary of the most recent dataframe.

    Call this when the user asks for a summary, explanation, or
    interpretation of the most recent result. If no dataframe is
    available, raise ModelRetry — the user is asking for a summary
    and deserves feedback.
    """
    df = ctx.deps.last_dataframe
    if df is None:
        raise ModelRetry(
            "No dataframe is available to summarize. Run a clinical-data tool "
            "or run_sql first, then call summarize_results."
        )
    if df.empty:
        raise ModelRetry(
            "The most recent result was empty; there is nothing to summarize. Re-run the query with different filters."
        )

    text = generate_summary_for_df(
        df=df,
        question=input.question,
        focus=input.focus,
    )

    if not text.strip():
        raise ModelRetry(
            "The summary backend could not generate a summary for this "
            "dataframe and question. Try rephrasing the question."
        )

    return SummaryArtifact(text=text)
