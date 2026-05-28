from __future__ import annotations
from datetime import date, datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from pydantic_ai import ModelRetry

from agent.deps import AgentDeps, SelectedPatient
from agent.db.analytics_adapter import AnalyticsDbAdapter


def _selected_john() -> SelectedPatient:
    return SelectedPatient(
        source_id="src-john-1962",
        display_name="John Smith",
        dob=date(1962, 5, 1),
        selected_at=datetime.now(),
        selection_origin="user_click",
    )


def _deps(synthetic_db, selected: SelectedPatient | None) -> AgentDeps:
    return AgentDeps(
        user_id=1,
        user_role=MagicMock(value=1),
        session_id="s1",
        selected_patient=selected,
        last_dataframe=None,
        last_sql=None,
        last_query_meta=None,
        analytics_db=AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite"),
        rag=None,
        sqlite_session=None,
        run_logger=MagicMock(),
    )


def test_make_chart_raises_model_retry_when_no_dataframe(synthetic_db):
    from agent.tools.make_chart import make_chart, MakeChartInput

    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, _selected_john())
    ctx.deps.last_dataframe = None

    with pytest.raises(ModelRetry, match="No dataframe"):
        make_chart(ctx, MakeChartInput(question="chart this"))


def test_make_chart_returns_chart_artifact_for_non_empty_df(synthetic_db):
    from agent.tools.make_chart import make_chart, MakeChartInput
    from agent.artifacts import ChartArtifact

    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, _selected_john())
    ctx.deps.last_dataframe = pd.DataFrame({"a": [1, 2, 3]})

    with patch("agent.tools.make_chart.generate_chart_for_df") as gen:
        gen.return_value = ('{"data":[]}', "code")
        result = make_chart(ctx, MakeChartInput(question="chart this"))

    assert isinstance(result, ChartArtifact)
    assert result.plotly_json == '{"data":[]}'
    assert result.chart_code == "code"


def test_make_chart_raises_model_retry_when_shim_returns_none(synthetic_db):
    from agent.tools.make_chart import make_chart, MakeChartInput

    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, _selected_john())
    ctx.deps.last_dataframe = pd.DataFrame({"a": [1]})

    with patch("agent.tools.make_chart.generate_chart_for_df") as gen:
        gen.return_value = (None, None)
        with pytest.raises(ModelRetry, match="could not generate"):
            make_chart(ctx, MakeChartInput(question="chart this"))
