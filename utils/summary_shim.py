"""Pure-function summary-generation shim.

Wraps Vanna's non-streaming generate_summary without side effects. The
agent's summarize_results tool calls this; streaming UX lives in the
runner's event loop, not the tool.
"""

from __future__ import annotations
from typing import Optional

import pandas as pd


def generate_summary_for_df(
    df: pd.DataFrame,
    question: str,
    focus: Optional[str] = None,
    vn=None,
) -> str:
    """Return a prose summary of `df` answering `question`.

    `focus` narrows the summary (e.g., "trend over time", "outliers").
    When provided, it's concatenated onto the question so the backend
    LLM sees both.

    Returns "" for empty DataFrames or when the backend returns None.
    """
    if df is None or df.empty:
        return ""

    if vn is None:
        from utils.chat_bot_helper import get_vn

        vn = get_vn()

    augmented_question = f"{question} (focus: {focus})" if focus else question
    text, _ = vn.generate_summary(question=augmented_question, df=df)
    return text or ""
