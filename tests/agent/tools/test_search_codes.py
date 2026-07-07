# tests/agent/tools/test_search_codes.py
"""Tests for agent.tools.search_codes, DB-backed via agent/codes/service.py
(Task 3 of the 2026-07-03 vocab port). Ported from chiron
tests/core/agent/tools/test_search_codes.py, adapted to thrive_ui's
AgentDeps/RunContext conventions and the `vocab_session` fixture in
tests/conftest.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from agent.deps import AgentDeps


def _deps(session) -> AgentDeps:
    return AgentDeps(
        user_id=1,
        user_role=MagicMock(value=1),
        session_id="s1",
        selected_patient=None,
        last_dataframe=None,
        last_sql=None,
        last_query_meta=None,
        analytics_db=None,
        rag=None,
        sqlite_session=session,
        run_logger=MagicMock(),
    )


def _ctx(session):
    ctx = MagicMock()
    ctx.deps = _deps(session)
    return ctx


def test_search_codes_returns_sets_first_for_condition_terms(vocab_session):
    from agent.tools.search_codes import CodeSearchInput, search_codes

    result = search_codes(_ctx(vocab_session), CodeSearchInput(vocabulary="icd10", query="DM"))
    assert result.sets and result.sets[0].set_id == "dx:diabetes-mellitus"
    assert result.sets[0].member_count == 2


def test_search_codes_codes_capped_at_limit(vocab_session):
    from agent.tools.search_codes import CodeSearchInput, search_codes

    result = search_codes(_ctx(vocab_session), CodeSearchInput(vocabulary="icd10", query="diabetes", limit=1))
    assert len(result.codes) <= 1


def test_search_codes_unloaded_vocab_has_actionable_error(vocab_session):
    """loinc has no rows in the fixture — VocabNotLoadedError must soft-fail
    into an empty result with an actionable note, not raise."""
    from agent.tools.search_codes import CodeSearchInput, search_codes

    result = search_codes(_ctx(vocab_session), CodeSearchInput(vocabulary="loinc", query="a1c"))
    assert result.note and "import_vocab_dump" in result.note
    assert result.codes == [] and result.sets == []


def test_search_codes_unknown_vocab_rejected():
    from agent.tools.search_codes import CodeSearchInput

    with pytest.raises(ValidationError):
        CodeSearchInput(vocabulary="unknown_vocab", query="x")


def test_search_codes_rejects_extra_fields():
    from agent.tools.search_codes import CodeSearchInput

    with pytest.raises(ValidationError):
        CodeSearchInput(vocabulary="icd10", query="x", totally_made_up_field="nope")


def test_search_codes_limit_over_50_rejected():
    from agent.tools.search_codes import CodeSearchInput

    with pytest.raises(ValidationError):
        CodeSearchInput(vocabulary="icd10", query="x", limit=51)


def test_search_codes_snomed_penicillin_allergy_end_to_end(vocab_session):
    """Epic #203 acceptance carried through the DB-backed rewrite: the
    allergy-intent lookup for 'penicillin allergy' still resolves via the
    synonym seeded onto the vocab_synonyms table in the vocab_session
    fixture (tests/conftest.py)."""
    from agent.tools.search_codes import CodeSearchInput, search_codes

    result = search_codes(_ctx(vocab_session), CodeSearchInput(vocabulary="snomed", query="penicillin allergy"))
    codes = {m.code for m in result.codes}
    assert "91936005" in codes, f"expected 91936005 in {codes}"
