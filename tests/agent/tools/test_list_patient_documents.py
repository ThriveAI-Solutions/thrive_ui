from datetime import date, datetime
from unittest.mock import MagicMock
import pytest

from agent.deps import AgentDeps, SelectedPatient
from agent.db.analytics_adapter import AnalyticsDbAdapter
from agent.tools.list_patient_documents import (
    list_patient_documents,
    DocumentIndexQuery,
    DocumentIndexResult,
    DocumentEntry,
)
from pydantic_ai import ModelRetry


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
        audit_logger=MagicMock(),
    )


def _john() -> SelectedPatient:
    return SelectedPatient(
        source_id="src-john-1962",
        display_name="John Smith",
        dob=date(1962, 5, 1),
        selected_at=datetime.now(),
        selection_origin="user_click",
    )


def test_list_documents_returns_three(synthetic_db):
    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, _john())
    result = list_patient_documents(ctx, DocumentIndexQuery())
    assert isinstance(result, DocumentIndexResult)
    assert result.data_availability == "data_present"
    assert len(result.documents) == 3
    assert all(isinstance(d, DocumentEntry) for d in result.documents)


def test_list_documents_carries_no_bodies_note(synthetic_db):
    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, _john())
    result = list_patient_documents(ctx, DocumentIndexQuery())
    assert "note bodies are not stored" in result.note.lower()


def test_list_documents_filtered_by_type(synthetic_db):
    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, _john())
    result = list_patient_documents(ctx, DocumentIndexQuery(document_type="radiology"))
    assert len(result.documents) == 1


def test_no_selection_raises_model_retry(synthetic_db):
    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, selected=None)
    with pytest.raises(ModelRetry, match="No patient is currently selected"):
        list_patient_documents(ctx, DocumentIndexQuery())


def test_list_patient_documents_sets_last_dataframe(synthetic_db):
    """After list_patient_documents returns, ctx.deps.last_dataframe
    should hold a pandas DataFrame of the result documents."""
    import pandas as pd

    ctx = MagicMock()
    ctx.deps = _deps(synthetic_db, _john())

    result = list_patient_documents(ctx, DocumentIndexQuery(date_range=None))

    assert isinstance(ctx.deps.last_dataframe, pd.DataFrame)
    assert len(ctx.deps.last_dataframe) == len(result.documents)
