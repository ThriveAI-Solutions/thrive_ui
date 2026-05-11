"""make_chart tool.

Phase 3 design §3.5 — operates on ctx.deps.last_dataframe. Wraps
utils.chart_shim.generate_chart_for_df (which wraps Vanna's helpers).
Returns a ChartArtifact for the runner to render through the existing
PLOTLY_CHART message type.
"""

from __future__ import annotations
from pydantic import BaseModel, Field
from pydantic_ai import RunContext
from pydantic_ai.exceptions import ModelRetry

from agent.artifacts import ChartArtifact
from agent.deps import AgentDeps
from utils.chart_shim import generate_chart_for_df


class MakeChartInput(BaseModel):
    question: str = Field(
        ...,
        description=(
            "What the chart should answer or emphasize, e.g. 'glucose values "
            "over time' or 'count of encounters per facility'. The chart "
            "code-generation model uses this to pick axes and chart type."
        ),
    )


def make_chart(ctx: RunContext[AgentDeps], input: MakeChartInput) -> ChartArtifact:
    """Generate a Plotly chart for the most recent dataframe on deps.

    Call this AFTER a clinical-data tool or run_sql has produced a
    dataframe. If no dataframe is available, raise ModelRetry — do
    NOT silently no-op; the user is asking for a chart and deserves
    feedback.
    """
    df = ctx.deps.last_dataframe
    if df is None:
        raise ModelRetry(
            "No dataframe is available to chart. Run a clinical-data tool or run_sql first, then call make_chart."
        )
    if df.empty:
        raise ModelRetry(
            "The most recent result was empty; there is nothing to chart. Re-run the query with different filters."
        )

    plotly_json, code = generate_chart_for_df(
        df=df,
        question=input.question,
        sql=ctx.deps.last_sql,
    )

    if plotly_json is None:
        raise ModelRetry(
            "The chart backend could not generate a chart for this dataframe "
            "and question. Try rephrasing the question or aggregating the "
            "data first."
        )

    return ChartArtifact(plotly_json=plotly_json, chart_code=code)
