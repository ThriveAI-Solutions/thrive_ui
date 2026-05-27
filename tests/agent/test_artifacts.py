"""Tests for the Phase 3 Artifact discriminated union.

Each variant carries the minimum payload its renderer needs. The union
is discriminated by `kind` so AgentResponse.artifacts can hold a mixed
list and the runtime can dispatch on `kind` without isinstance chains.
"""

from __future__ import annotations
import json
import pytest
from pydantic import TypeAdapter, ValidationError

from agent.artifacts import (
    Artifact,
    ChartArtifact,
    DataFrameArtifact,
    SqlArtifact,
    SummaryArtifact,
)


def test_chart_artifact_round_trip():
    a = ChartArtifact(plotly_json='{"data":[]}', chart_code="print(1)")
    dumped = a.model_dump(mode="json")
    assert dumped == {
        "kind": "chart",
        "plotly_json": '{"data":[]}',
        "chart_code": "print(1)",
    }
    reloaded = ChartArtifact.model_validate(dumped)
    assert reloaded == a


def test_summary_artifact_round_trip():
    a = SummaryArtifact(text="Five rows. Means within range.")
    assert a.model_dump() == {"kind": "summary", "text": "Five rows. Means within range."}


def test_dataframe_artifact_round_trip():
    a = DataFrameArtifact(
        columns=["a", "b"],
        rows=[[1, 2], [3, 4]],
        row_count=2,
        truncated=False,
    )
    assert a.row_count == 2
    assert a.truncated is False


def test_sql_artifact_round_trip():
    a = SqlArtifact(sql="SELECT 1")
    assert a.model_dump() == {"kind": "sql", "sql": "SELECT 1"}


def test_union_dispatches_on_kind():
    adapter = TypeAdapter(Artifact)
    chart = adapter.validate_python({"kind": "chart", "plotly_json": "{}"})
    summary = adapter.validate_python({"kind": "summary", "text": "x"})
    df = adapter.validate_python({"kind": "dataframe", "columns": ["c"], "rows": [["v"]], "row_count": 1})
    sql = adapter.validate_python({"kind": "sql", "sql": "SELECT 1"})

    assert isinstance(chart, ChartArtifact)
    assert isinstance(summary, SummaryArtifact)
    assert isinstance(df, DataFrameArtifact)
    assert isinstance(sql, SqlArtifact)


def test_union_rejects_unknown_kind():
    adapter = TypeAdapter(Artifact)
    with pytest.raises(ValidationError):
        adapter.validate_python({"kind": "spaghetti", "wat": True})


def test_dataframe_artifact_truncated_default_false():
    a = DataFrameArtifact(columns=["a"], rows=[[1]], row_count=1)
    assert a.truncated is False
