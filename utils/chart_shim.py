"""Pure-function chart-generation shim.

Wraps the existing Vanna chart helpers from utils.chat_bot_helper.get_chart
without side effects on the Streamlit message stream. The agent's
make_chart tool calls this; the chat handler renders the resulting
ChartArtifact through the existing PLOTLY_CHART renderer.
"""

from __future__ import annotations
from typing import Optional, Tuple

import pandas as pd


def generate_chart_for_df(
    df: pd.DataFrame,
    question: str,
    sql: Optional[str] = None,
    vn=None,
) -> Tuple[Optional[str], Optional[str]]:
    """Generate a Plotly chart for `df` answering `question`.

    Returns (plotly_json, chart_code). Either may be None:
      - both None if the DataFrame is empty or the backend declines.
      - plotly_json None but code present if the code didn't produce a figure.

    `vn` is the Vanna service; defaults to chat_bot_helper.get_vn() for
    convenience. Tests inject a mock.
    """
    if df is None or df.empty:
        return None, None

    if vn is None:
        from utils.chat_bot_helper import get_vn

        vn = get_vn()

    if not vn.should_generate_chart(question=question, sql=sql, df=df):
        return None, None

    code_result = vn.generate_plotly_code(question=question, sql=sql, df=df)
    code, _ = code_result if isinstance(code_result, tuple) else (code_result, 0)

    if not code:
        return None, None

    plot_result = vn.generate_plot(code=code, df=df)
    fig, _ = plot_result if isinstance(plot_result, tuple) else (plot_result, 0)

    if fig is None:
        return None, code

    return fig.to_json(), code
