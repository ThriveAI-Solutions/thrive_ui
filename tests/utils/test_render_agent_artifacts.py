"""When an AgentResponse carries SQL / DataFrame artifacts, the chat-render
pipeline should emit corresponding MessageType.SQL and MessageType.DATAFRAME
messages so the existing MESSAGE_RENDERERS dispatch fires.

Per Phase 3 design §3.3 — artifacts dispatch through existing renderers
with no new renderer code.
"""

from __future__ import annotations
from unittest.mock import patch

import pandas as pd
import pytest

from agent.artifacts import (
    ChartArtifact,
    DataFrameArtifact,
    SqlArtifact,
    SummaryArtifact,
)
from agent.state import AgentResponse


def test_render_artifacts_emits_sql_message_for_sql_artifact():
    from utils.chat_bot_helper import render_agent_artifacts

    response = AgentResponse(
        text="here's the result",
        artifacts=[SqlArtifact(sql="SELECT 1")],
    )

    emitted: list = []
    with patch("utils.chat_bot_helper.add_message", side_effect=emitted.append):
        render_agent_artifacts(response, question="ad-hoc")

    assert len(emitted) == 1
    msg = emitted[0]
    assert msg.type == "sql"
    assert msg.content == "SELECT 1"


def test_render_artifacts_emits_dataframe_message_for_dataframe_artifact():
    from utils.chat_bot_helper import render_agent_artifacts

    response = AgentResponse(
        text="rows",
        artifacts=[
            DataFrameArtifact(
                columns=["a", "b"],
                rows=[[1, "x"], [2, "y"]],
                row_count=2,
                truncated=False,
            )
        ],
    )

    emitted: list = []
    with patch("utils.chat_bot_helper.add_message", side_effect=emitted.append):
        render_agent_artifacts(response, question="ad-hoc")

    assert len(emitted) == 1
    assert emitted[0].type == "dataframe"


def test_render_artifacts_handles_multiple_variants():
    from utils.chat_bot_helper import render_agent_artifacts

    response = AgentResponse(
        text="both",
        artifacts=[
            SqlArtifact(sql="SELECT 1"),
            DataFrameArtifact(columns=["a"], rows=[[1]], row_count=1),
        ],
    )

    emitted: list = []
    with patch("utils.chat_bot_helper.add_message", side_effect=emitted.append):
        render_agent_artifacts(response, question="ad-hoc")

    assert [m.type for m in emitted] == ["sql", "dataframe"]


def test_render_artifacts_emits_plotly_chart_for_chart_artifact():
    from utils.chat_bot_helper import render_agent_artifacts
    from agent.state import AgentResponse
    from agent.artifacts import ChartArtifact
    from unittest.mock import patch

    response = AgentResponse(
        text="here's a chart",
        artifacts=[ChartArtifact(plotly_json='{"data":[],"layout":{}}', chart_code="code")],
    )

    emitted: list = []
    with patch("utils.chat_bot_helper.add_message", side_effect=emitted.append):
        render_agent_artifacts(response, question="chart this")

    assert len(emitted) == 1
    # Message.type is stored as a string (the enum value)
    assert emitted[0].type == "plotly_chart"


def test_render_artifacts_emits_summary_for_summary_artifact():
    from utils.chat_bot_helper import render_agent_artifacts
    from agent.state import AgentResponse
    from agent.artifacts import SummaryArtifact
    from unittest.mock import patch

    response = AgentResponse(
        text="here's a summary",
        artifacts=[SummaryArtifact(text="five rows; mean=3")],
    )

    emitted: list = []
    with patch("utils.chat_bot_helper.add_message", side_effect=emitted.append):
        render_agent_artifacts(response, question="describe")

    assert len(emitted) == 1
    # Message.type is stored as a string (the enum value)
    assert emitted[0].type == "summary"
    assert emitted[0].content == "five rows; mean=3"
