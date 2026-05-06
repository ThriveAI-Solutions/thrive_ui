import pytest
from datetime import date
from unittest.mock import MagicMock
from agent.deps import AgentDeps
from agent.db.analytics_adapter import AnalyticsDbAdapter
from agent.tools.find_patient import (
    find_patient,
    PatientSearchQuery,
    PatientSearchResults,
)


@pytest.fixture
def deps_factory(synthetic_db):
    def _make() -> AgentDeps:
        adapter = AnalyticsDbAdapter(engine=synthetic_db, dialect="sqlite")
        return AgentDeps(
            user_id=1,
            user_role=MagicMock(value=1),
            session_id="s1",
            selected_patient=None,
            last_dataframe=None,
            last_sql=None,
            last_query_meta=None,
            analytics_db=adapter,
            rag=None,
            sqlite_session=None,
            audit_logger=MagicMock(),
        )

    return _make


def test_find_patient_returns_three_smith_results(deps_factory):
    ctx = MagicMock()
    ctx.deps = deps_factory()
    q = PatientSearchQuery(last_name="Smith")
    result = find_patient(ctx, q)
    assert isinstance(result, PatientSearchResults)
    assert result.total_unique == 3
    source_ids = sorted(m.source_id for m in result.matches)
    assert source_ids == ["src-jane-1985", "src-john-1962", "src-john-1971"]


def test_find_patient_excludes_rank_99(deps_factory):
    ctx = MagicMock()
    ctx.deps = deps_factory()
    result = find_patient(ctx, PatientSearchQuery(last_name="Smith"))
    for m in result.matches:
        assert m.source_id != "src-john-1962-stale"


def test_find_patient_includes_related_source_ids(deps_factory):
    ctx = MagicMock()
    ctx.deps = deps_factory()
    result = find_patient(ctx, PatientSearchQuery(first_name="John", last_name="Smith"))
    john_1962 = next(m for m in result.matches if m.source_id == "src-john-1962")
    assert "src-john-1962-alt" in john_1962.related_source_ids


def test_find_patient_query_validation():
    with pytest.raises(Exception):
        PatientSearchQuery(limit=200)  # over le=100


def test_find_patient_no_results_returns_empty(deps_factory):
    ctx = MagicMock()
    ctx.deps = deps_factory()
    result = find_patient(ctx, PatientSearchQuery(last_name="NoSuchName"))
    assert result.total_unique == 0
    assert result.matches == []
