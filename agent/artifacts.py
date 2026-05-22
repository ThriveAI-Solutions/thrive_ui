"""Discriminated-union artifact shapes for AgentResponse.artifacts.

Each variant maps 1:1 to an existing utils.enums.MessageType so the
chat-render pipeline dispatches through MESSAGE_RENDERERS without any
new renderer code:

    ChartArtifact     → MessageType.PLOTLY_CHART
    SummaryArtifact   → MessageType.SUMMARY
    DataFrameArtifact → MessageType.DATAFRAME
    SqlArtifact       → MessageType.SQL

Per Phase 3 design §3.3 and parent spec §7.10.
"""

from __future__ import annotations
from typing import Annotated, Any, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class ChartArtifact(BaseModel):
    kind: Literal["chart"] = "chart"
    plotly_json: str
    chart_code: Optional[str] = None


class SummaryArtifact(BaseModel):
    kind: Literal["summary"] = "summary"
    text: str


class DataFrameArtifact(BaseModel):
    kind: Literal["dataframe"] = "dataframe"
    columns: List[str]
    rows: List[List[Any]]
    row_count: int
    truncated: bool = False


class SqlArtifact(BaseModel):
    kind: Literal["sql"] = "sql"
    sql: str


Artifact = Annotated[
    Union[ChartArtifact, SummaryArtifact, DataFrameArtifact, SqlArtifact],
    Field(discriminator="kind"),
]
