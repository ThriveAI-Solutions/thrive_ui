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


def test_summarize_raises_model_retry_when_no_dataframe(synthetic_db):
    from agent.tools.summarize_results import summarize_results, SummarizeResultsInput

    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, _selected_john())
    ctx.deps.last_dataframe = None

    with pytest.raises(ModelRetry, match="No dataframe"):
        summarize_results(ctx, SummarizeResultsInput(question="describe"))


def test_summarize_returns_summary_artifact(synthetic_db):
    from agent.tools.summarize_results import summarize_results, SummarizeResultsInput
    from agent.artifacts import SummaryArtifact

    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, _selected_john())
    ctx.deps.last_dataframe = pd.DataFrame({"a": [1, 2]})

    with patch("agent.tools.summarize_results.generate_summary_for_df") as gen:
        gen.return_value = "two rows; mean=1.5"
        result = summarize_results(ctx, SummarizeResultsInput(question="describe"))

    assert isinstance(result, SummaryArtifact)
    assert result.text == "two rows; mean=1.5"


def test_summarize_passes_focus_through(synthetic_db):
    from agent.tools.summarize_results import summarize_results, SummarizeResultsInput

    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, _selected_john())
    ctx.deps.last_dataframe = pd.DataFrame({"a": [1]})

    with patch("agent.tools.summarize_results.generate_summary_for_df") as gen:
        gen.return_value = "ok"
        summarize_results(
            ctx,
            SummarizeResultsInput(question="describe", focus="outliers"),
        )

    assert gen.call_args.kwargs.get("focus") == "outliers"


def test_summarize_raises_model_retry_when_summary_empty(synthetic_db):
    from agent.tools.summarize_results import summarize_results, SummarizeResultsInput

    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, _selected_john())
    ctx.deps.last_dataframe = pd.DataFrame({"a": [1]})

    with patch("agent.tools.summarize_results.generate_summary_for_df") as gen:
        gen.return_value = ""
        with pytest.raises(ModelRetry, match="could not generate"):
            summarize_results(ctx, SummarizeResultsInput(question="describe"))
