"""Tests for the Scope classification + filter on the Question audit page.

Epic #166 / Feature #167. Classification is computed in SQL via a CASE
expression in ``_question_audit_base_query`` so that pagination (``COUNT(*)``
+ ``OFFSET/LIMIT``) stays honest under the Scope filter — see Epic #166
Architecture Considerations.

The four buckets:

  - ``Patient``      slot filled OR any patient-intent tool called
  - ``Pop Health``   not Patient AND ``search_patients_by_criteria`` called
  - ``Other``        AgentRun row exists, neither above
  - ``Legacy/Unknown`` no AgentRun row at all
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from orm.models import AgentRun, Base, Message, RoleTypeEnum, ToolCall, User, UserRole
from utils.enums import MessageType, RoleType


# ---------------------------------------------------------------------------
# Fixtures + helpers (lightly adapted from tests/orm/test_question_audit.py)
# ---------------------------------------------------------------------------


@pytest.fixture
def session_factory(monkeypatch):
    """In-memory SQLite with SessionLocal patched on orm.logging_functions."""
    from orm import logging_functions as lf

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    monkeypatch.setattr(lf, "SessionLocal", Session)
    return Session


def _seed_roles(session):
    session.add_all(
        [
            UserRole(id=1, role_name="Admin", description="Admin", role=RoleTypeEnum.ADMIN),
            UserRole(id=4, role_name="Patient", description="Patient", role=RoleTypeEnum.PATIENT),
        ]
    )
    session.commit()


def _seed_user(session, *, id, username, org=None, role_id=4):
    session.add(
        User(
            id=id,
            user_role_id=role_id,
            username=username,
            first_name=username,
            last_name="X",
            password="hash:abc",
            email=f"{username}@x.io",
            organization=org,
        )
    )
    session.commit()


def _add_question(session, *, user_id, content, when):
    """Seed a user-role Message row with a clamped ``created_at``. Returns the
    new Message.id so the caller can link an AgentRun to it.
    """
    session.add(
        Message(
            role=RoleType.USER,
            content=content,
            type=MessageType.TEXT,
            user_id=user_id,
        )
    )
    session.commit()
    last = session.query(Message).order_by(Message.id.desc()).first()
    last.created_at = when
    session.commit()
    return last.id


_DEFAULT_TOOL_KW = dict(
    arguments_json="{}",
    result_summary="{}",
    elapsed_ms=1,
    success=True,
)


def _add_agent_run(
    session,
    *,
    run_id,
    user_id,
    user_message_id,
    selected_patient_source_id=None,
):
    """Seed a minimal AgentRun row keyed to ``user_message_id``."""
    session.add(
        AgentRun(
            run_id=run_id,
            session_id=f"sess-{run_id}",
            user_id=user_id,
            user_role=RoleTypeEnum.PATIENT.value,
            user_message_id=user_message_id,
            selected_patient_source_id=selected_patient_source_id,
            tool_call_count=0,
            event_count=0,
            status="completed",
            success=True,
            review_status="unreviewed",
            logging_mode="full",
            schema_version=1,
        )
    )
    session.commit()


def _add_tool_call(session, *, run_id, user_id, tool_name):
    session.add(
        ToolCall(
            session_id=f"sess-{run_id}",
            user_id=user_id,
            user_role=RoleTypeEnum.PATIENT.value,
            tool_name=tool_name,
            run_id=run_id,
            **_DEFAULT_TOOL_KW,
        )
    )
    session.commit()


def _filters(**overrides):
    base = {"usernames": [], "orgs": [], "days": 30, "search": None}
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# (1) Classification correctness — one row per bucket
# ---------------------------------------------------------------------------


def test_scope_legacy_when_no_agent_run(session_factory):
    """A user-role Message with no linked AgentRun row classifies as
    ``Legacy/Unknown`` (the pre-agentic-replatform case)."""
    from orm.logging_functions import get_question_audit_page

    s = session_factory()
    _seed_roles(s)
    _seed_user(s, id=10, username="alice")
    _add_question(s, user_id=10, content="legacy q", when=datetime(2026, 6, 1, 12, 0, 0))

    page = get_question_audit_page(_filters(), page=1, page_size=50)
    assert len(page["items"]) == 1
    assert page["items"][0]["scope"] == "Legacy/Unknown"


def test_scope_patient_when_slot_filled(session_factory):
    """``AgentRun.selected_patient_source_id IS NOT NULL`` => Patient."""
    from orm.logging_functions import get_question_audit_page

    s = session_factory()
    _seed_roles(s)
    _seed_user(s, id=10, username="alice")
    mid = _add_question(s, user_id=10, content="slot-filled", when=datetime(2026, 6, 1, 12, 0, 0))
    _add_agent_run(s, run_id="run-a", user_id=10, user_message_id=mid, selected_patient_source_id="pat-123")

    page = get_question_audit_page(_filters(), page=1, page_size=50)
    assert page["items"][0]["scope"] == "Patient"


@pytest.mark.parametrize(
    "tool_name",
    ["find_patient", "get_patient_clinical_data", "list_patient_documents"],
)
def test_scope_patient_when_patient_intent_tool_only(session_factory, tool_name):
    """Patient *intent* counts even when the slot never filled — a run that
    called e.g. ``find_patient`` without picking a candidate still classifies
    as Patient. This is the edge case spelled out in the spec."""
    from orm.logging_functions import get_question_audit_page

    s = session_factory()
    _seed_roles(s)
    _seed_user(s, id=10, username="alice")
    mid = _add_question(s, user_id=10, content=f"intent-{tool_name}", when=datetime(2026, 6, 1, 12, 0, 0))
    _add_agent_run(s, run_id="run-b", user_id=10, user_message_id=mid)  # slot empty
    _add_tool_call(s, run_id="run-b", user_id=10, tool_name=tool_name)

    page = get_question_audit_page(_filters(), page=1, page_size=50)
    assert page["items"][0]["scope"] == "Patient"


def test_scope_pop_health(session_factory):
    """Not Patient AND ``search_patients_by_criteria`` was called => Pop Health."""
    from orm.logging_functions import get_question_audit_page

    s = session_factory()
    _seed_roles(s)
    _seed_user(s, id=10, username="alice")
    mid = _add_question(s, user_id=10, content="cohort q", when=datetime(2026, 6, 1, 12, 0, 0))
    _add_agent_run(s, run_id="run-c", user_id=10, user_message_id=mid)  # no slot
    _add_tool_call(s, run_id="run-c", user_id=10, tool_name="search_patients_by_criteria")

    page = get_question_audit_page(_filters(), page=1, page_size=50)
    assert page["items"][0]["scope"] == "Pop Health"


def test_scope_other_when_agent_run_but_no_classified_tools(session_factory):
    """AgentRun exists, no slot, no Patient/Pop tools => Other."""
    from orm.logging_functions import get_question_audit_page

    s = session_factory()
    _seed_roles(s)
    _seed_user(s, id=10, username="alice")
    mid = _add_question(s, user_id=10, content="kb only", when=datetime(2026, 6, 1, 12, 0, 0))
    _add_agent_run(s, run_id="run-d", user_id=10, user_message_id=mid)
    _add_tool_call(s, run_id="run-d", user_id=10, tool_name="search_knowledge_base")
    _add_tool_call(s, run_id="run-d", user_id=10, tool_name="run_sql")

    page = get_question_audit_page(_filters(), page=1, page_size=50)
    assert page["items"][0]["scope"] == "Other"


def test_scope_patient_wins_over_pop_when_both_tool_families_fire(session_factory):
    """When a single run fires both a Patient-intent tool and the Pop Health
    tool, the spec's order says Patient wins. This is the same evaluation
    order the SQL ``CASE`` uses."""
    from orm.logging_functions import get_question_audit_page

    s = session_factory()
    _seed_roles(s)
    _seed_user(s, id=10, username="alice")
    mid = _add_question(s, user_id=10, content="mixed", when=datetime(2026, 6, 1, 12, 0, 0))
    _add_agent_run(s, run_id="run-e", user_id=10, user_message_id=mid)
    _add_tool_call(s, run_id="run-e", user_id=10, tool_name="find_patient")
    _add_tool_call(s, run_id="run-e", user_id=10, tool_name="search_patients_by_criteria")

    page = get_question_audit_page(_filters(), page=1, page_size=50)
    assert page["items"][0]["scope"] == "Patient"


# ---------------------------------------------------------------------------
# (2) Filter + total + pagination correctness on a mixed page
# ---------------------------------------------------------------------------


def _seed_mixed_page(s) -> dict[str, int]:
    """Seed 10 user messages spread across all 4 buckets. Returns a dict
    ``{bucket: count}`` for cross-checking expectations.

    Counts: 4 Patient · 2 Pop Health · 2 Other · 2 Legacy/Unknown.
    Timestamps are spaced so reverse-chronological order is deterministic.
    """
    _seed_roles(s)
    _seed_user(s, id=10, username="alice")
    base = datetime(2026, 6, 1, 12, 0, 0)
    n = 0

    def _ts(i):
        # Newest row first (i=0 is newest).
        return base - timedelta(minutes=i)

    # Patient rows
    for i in range(4):
        mid = _add_question(s, user_id=10, content=f"patient-{i}", when=_ts(n))
        _add_agent_run(s, run_id=f"r-pat-{i}", user_id=10, user_message_id=mid, selected_patient_source_id=f"p{i}")
        n += 1

    # Pop Health rows
    for i in range(2):
        mid = _add_question(s, user_id=10, content=f"pop-{i}", when=_ts(n))
        _add_agent_run(s, run_id=f"r-pop-{i}", user_id=10, user_message_id=mid)
        _add_tool_call(s, run_id=f"r-pop-{i}", user_id=10, tool_name="search_patients_by_criteria")
        n += 1

    # Other rows
    for i in range(2):
        mid = _add_question(s, user_id=10, content=f"other-{i}", when=_ts(n))
        _add_agent_run(s, run_id=f"r-oth-{i}", user_id=10, user_message_id=mid)
        _add_tool_call(s, run_id=f"r-oth-{i}", user_id=10, tool_name="search_knowledge_base")
        n += 1

    # Legacy rows — no AgentRun
    for i in range(2):
        _add_question(s, user_id=10, content=f"legacy-{i}", when=_ts(n))
        n += 1

    assert n == 10
    return {"Patient": 4, "Pop Health": 2, "Other": 2, "Legacy/Unknown": 2}


def test_scope_filter_reduces_total_and_items(session_factory):
    """Filtering by Scope must shrink BOTH ``total`` (via SQL WHERE before
    COUNT(*)) and the returned ``items``."""
    from orm.logging_functions import get_question_audit_page

    s = session_factory()
    expected = _seed_mixed_page(s)

    # Unfiltered: 10 rows.
    unfiltered = get_question_audit_page(_filters(), page=1, page_size=50)
    assert unfiltered["total"] == 10
    assert len(unfiltered["items"]) == 10

    # Filtered to Patient only.
    only_patient = get_question_audit_page(_filters(scopes=["Patient"]), page=1, page_size=50)
    assert only_patient["total"] == expected["Patient"]
    assert {it["scope"] for it in only_patient["items"]} == {"Patient"}
    assert len(only_patient["items"]) == expected["Patient"]


def test_scope_filter_multiple_buckets_union(session_factory):
    """A multi-bucket filter is OR-style (multiselect)."""
    from orm.logging_functions import get_question_audit_page

    s = session_factory()
    expected = _seed_mixed_page(s)

    result = get_question_audit_page(_filters(scopes=["Pop Health", "Legacy/Unknown"]), page=1, page_size=50)
    assert result["total"] == expected["Pop Health"] + expected["Legacy/Unknown"]
    assert {it["scope"] for it in result["items"]} == {"Pop Health", "Legacy/Unknown"}


def test_scope_filter_pagination_slices_filtered_set(session_factory):
    """Page 2 under a Scope filter must return the correct slice of the
    FILTERED set, not the unfiltered set. With page_size=2 + Patient filter
    (4 rows), page 1 has 2 Patient rows and page 2 has the remaining 2."""
    from orm.logging_functions import get_question_audit_page

    s = session_factory()
    _seed_mixed_page(s)

    p1 = get_question_audit_page(_filters(scopes=["Patient"]), page=1, page_size=2)
    p2 = get_question_audit_page(_filters(scopes=["Patient"]), page=2, page_size=2)

    assert p1["total"] == 4
    assert p2["total"] == 4
    assert len(p1["items"]) == 2
    assert len(p2["items"]) == 2
    # All four are Patient.
    assert {it["scope"] for it in (*p1["items"], *p2["items"])} == {"Patient"}
    # Disjoint slices.
    p1_ids = {it["user_message_id"] for it in p1["items"]}
    p2_ids = {it["user_message_id"] for it in p2["items"]}
    assert p1_ids.isdisjoint(p2_ids)


def test_scope_filter_empty_list_means_no_filter(session_factory):
    """Empty / None scopes must NOT filter — same shape as the existing
    usernames / orgs filters."""
    from orm.logging_functions import get_question_audit_page

    s = session_factory()
    _seed_mixed_page(s)

    none_filter = get_question_audit_page(_filters(scopes=None), page=1, page_size=50)
    empty_filter = get_question_audit_page(_filters(scopes=[]), page=1, page_size=50)
    assert none_filter["total"] == 10
    assert empty_filter["total"] == 10


def test_scope_filter_unknown_bucket_is_ignored(session_factory):
    """Defensive: an unrecognised scope label is silently dropped from the
    filter set, so the rest still narrow the result. (If we let unknowns
    through to ``IN (...)`` the user sees an empty page instead of just
    their intended filter.)"""
    from orm.logging_functions import get_question_audit_page

    s = session_factory()
    expected = _seed_mixed_page(s)

    result = get_question_audit_page(_filters(scopes=["Patient", "BogusBucket"]), page=1, page_size=50)
    assert result["total"] == expected["Patient"]


# ---------------------------------------------------------------------------
# (3) CSV export honors the filter + classification flows through
# ---------------------------------------------------------------------------


def test_export_respects_scope_filter_and_carries_scope_field(session_factory):
    from orm.logging_functions import get_question_audit_export

    s = session_factory()
    expected = _seed_mixed_page(s)

    rows = get_question_audit_export(_filters(scopes=["Pop Health"]))
    assert len(rows) == expected["Pop Health"]
    assert {r["scope"] for r in rows} == {"Pop Health"}
    # Same item shape as the page items — the existing keys must survive.
    assert "question" in rows[0]
    assert "status" in rows[0]
