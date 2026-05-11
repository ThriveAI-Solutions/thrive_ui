"""Pure-function shim for chart generation.

Phase 3 design §3.5 — the agent's make_chart tool wraps the existing
Vanna chart-gen helpers but does NOT add messages to the chat (that's
the runner's job after make_chart returns an artifact).

The shim takes (df, question, sql?) and returns (plotly_json, code).
"""

from __future__ import annotations
from unittest.mock import MagicMock

import pandas as pd
import pytest


def test_generate_chart_returns_plotly_json_and_code():
    from utils.chart_shim import generate_chart_for_df

    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    fake_vn = MagicMock()
    fake_vn.should_generate_chart.return_value = True
    fake_vn.generate_plotly_code.return_value = ("print('hi')", 0.1)
    fake_fig = MagicMock()
    fake_fig.to_json.return_value = '{"data":[],"layout":{}}'
    fake_vn.generate_plot.return_value = (fake_fig, 0.2)

    plotly_json, code = generate_chart_for_df(df, question="show", vn=fake_vn)

    assert plotly_json == '{"data":[],"layout":{}}'
    assert code == "print('hi')"
    fake_vn.generate_plotly_code.assert_called_once()
    fake_vn.generate_plot.assert_called_once()


def test_generate_chart_returns_none_when_should_not_chart():
    from utils.chart_shim import generate_chart_for_df

    df = pd.DataFrame({"a": [1]})
    fake_vn = MagicMock()
    fake_vn.should_generate_chart.return_value = False

    plotly_json, code = generate_chart_for_df(df, question="show", vn=fake_vn)

    assert plotly_json is None
    assert code is None
    fake_vn.generate_plotly_code.assert_not_called()


def test_generate_chart_returns_none_on_empty_dataframe():
    from utils.chart_shim import generate_chart_for_df

    df = pd.DataFrame()
    fake_vn = MagicMock()

    plotly_json, code = generate_chart_for_df(df, question="show", vn=fake_vn)

    assert plotly_json is None
    assert code is None
    fake_vn.should_generate_chart.assert_not_called()


def test_generate_chart_passes_sql_when_provided():
    from utils.chart_shim import generate_chart_for_df

    df = pd.DataFrame({"a": [1]})
    fake_vn = MagicMock()
    fake_vn.should_generate_chart.return_value = True
    fake_vn.generate_plotly_code.return_value = ("code", 0)
    fake_fig = MagicMock()
    fake_fig.to_json.return_value = "{}"
    fake_vn.generate_plot.return_value = (fake_fig, 0)

    generate_chart_for_df(df, question="q", sql="SELECT 1", vn=fake_vn)

    call_args = fake_vn.generate_plotly_code.call_args
    assert call_args.kwargs.get("sql") == "SELECT 1"
