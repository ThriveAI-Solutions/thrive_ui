"""Pure-function shim for non-streaming summary generation.

The agent's summarize_results tool wraps this. We use the non-streaming
variant inside the tool because tools must return a value; streaming
UX inside agent mode is the runner's job, not the tool's.
"""

from __future__ import annotations
from unittest.mock import MagicMock

import pandas as pd


def test_generate_summary_returns_text():
    from utils.summary_shim import generate_summary_for_df

    df = pd.DataFrame({"a": [1, 2, 3]})
    fake_vn = MagicMock()
    fake_vn.generate_summary.return_value = ("three rows; mean=2", 0.5)

    text = generate_summary_for_df(df, question="describe", vn=fake_vn)

    assert text == "three rows; mean=2"
    fake_vn.generate_summary.assert_called_once_with(question="describe", df=df)


def test_generate_summary_returns_empty_string_for_empty_df():
    from utils.summary_shim import generate_summary_for_df

    df = pd.DataFrame()
    fake_vn = MagicMock()

    text = generate_summary_for_df(df, question="describe", vn=fake_vn)

    assert text == ""
    fake_vn.generate_summary.assert_not_called()


def test_generate_summary_returns_empty_when_backend_returns_none():
    from utils.summary_shim import generate_summary_for_df

    df = pd.DataFrame({"a": [1]})
    fake_vn = MagicMock()
    fake_vn.generate_summary.return_value = (None, 0.0)

    text = generate_summary_for_df(df, question="describe", vn=fake_vn)

    assert text == ""


def test_generate_summary_applies_focus_to_question():
    from utils.summary_shim import generate_summary_for_df

    df = pd.DataFrame({"a": [1]})
    fake_vn = MagicMock()
    fake_vn.generate_summary.return_value = ("text", 0.0)

    generate_summary_for_df(df, question="describe", focus="trend over time", vn=fake_vn)

    call_args = fake_vn.generate_summary.call_args
    assert "trend over time" in call_args.kwargs["question"]
