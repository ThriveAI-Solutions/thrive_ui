"""Tests for orm.logging_functions question-audit helpers (#134)."""

from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from orm.models import Base, Message, RoleTypeEnum, User, UserRole
from utils.enums import MessageType, RoleType


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


def _seed_user(session, *, id, username, org="TestOrg", role_id=4):
    # NOTE: post Epic #179 `organization` is NOT NULL. Callers that pass
    # ``org=None`` to exercise the legacy "(no org)" sentinel filter are
    # now incompatible with the schema — those tests are skipped below.
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


def _add_question(session, *, user_id, content, when, elapsed=None, sql=None, summary=None, dataframe=None, error=None):
    """Seed a user message + optional assistant rows (SQL / summary / dataframe / error)."""
    session.add(
        Message(
            role=RoleType.USER,
            content=content,
            type=MessageType.TEXT,
            user_id=user_id,
        )
    )
    session.commit()
    # Override created_at after insert (server_default=func.now() is set on insert)
    last = session.query(Message).order_by(Message.id.desc()).first()
    last.created_at = when
    session.commit()

    if sql is not None:
        session.add(
            Message(
                role=RoleType.ASSISTANT,
                content=sql,
                type=MessageType.SQL,
                user_id=user_id,
                question=content,
                elapsed_time=elapsed,
            )
        )
    if summary is not None:
        session.add(
            Message(
                role=RoleType.ASSISTANT,
                content=summary,
                type=MessageType.SUMMARY,
                user_id=user_id,
                question=content,
            )
        )
    if dataframe is not None:
        session.add(
            Message(
                role=RoleType.ASSISTANT,
                content=dataframe,
                type=MessageType.DATAFRAME,
                user_id=user_id,
                question=content,
            )
        )
    if error is not None:
        session.add(
            Message(
                role=RoleType.ASSISTANT,
                content=error,
                type=MessageType.ERROR,
                user_id=user_id,
                question=content,
            )
        )
    session.commit()


def _filters(usernames=None, orgs=None, days=30, search=None):
    return {"usernames": usernames or [], "orgs": orgs or [], "days": days, "search": search}


def test_empty_database_returns_empty(session_factory):
    from orm.logging_functions import get_question_audit_filter_options, get_question_audit_page

    s = session_factory()
    _seed_roles(s)

    page = get_question_audit_page(_filters(), page=1, page_size=50)
    assert page == {"items": [], "total": 0}

    opts = get_question_audit_filter_options(days=30)
    assert opts == {"usernames": [], "orgs": []}


def test_single_user_three_questions_reverse_chrono(session_factory):
    from orm.logging_functions import get_question_audit_page

    s = session_factory()
    _seed_roles(s)
    _seed_user(s, id=10, username="alice", org="Acme")

    now = datetime(2026, 6, 1, 12, 0, 0)
    _add_question(s, user_id=10, content="q1 oldest", when=now - timedelta(hours=2))
    _add_question(s, user_id=10, content="q2 middle", when=now - timedelta(hours=1))
    _add_question(s, user_id=10, content="q3 newest", when=now)

    result = get_question_audit_page(_filters(), page=1, page_size=50)
    qs = [it["question"] for it in result["items"]]
    assert qs == ["q3 newest", "q2 middle", "q1 oldest"]
    assert result["total"] == 3


def test_pagination(session_factory):
    from orm.logging_functions import get_question_audit_page

    s = session_factory()
    _seed_roles(s)
    _seed_user(s, id=10, username="alice")
    base = datetime(2026, 6, 1, 12, 0, 0)
    for i in range(3):
        _add_question(s, user_id=10, content=f"q{i}", when=base - timedelta(hours=i))

    p1 = get_question_audit_page(_filters(), page=1, page_size=2)
    p2 = get_question_audit_page(_filters(), page=2, page_size=2)
    assert len(p1["items"]) == 2
    assert len(p2["items"]) == 1
    assert p1["total"] == 3 and p2["total"] == 3


def test_username_filter(session_factory):
    from orm.logging_functions import get_question_audit_page

    s = session_factory()
    _seed_roles(s)
    _seed_user(s, id=10, username="alice")
    _seed_user(s, id=11, username="bob")
    now = datetime(2026, 6, 1, 12, 0, 0)
    _add_question(s, user_id=10, content="a-only", when=now)
    _add_question(s, user_id=11, content="b-only", when=now)

    result = get_question_audit_page(_filters(usernames=["alice"]), page=1, page_size=50)
    qs = [it["question"] for it in result["items"]]
    assert qs == ["a-only"]


@pytest.mark.skip(
    reason="Epic #179: organization is now NOT NULL on thrive_user, so the legacy "
    "'(no org)' sentinel filter has no rows to match. Existing NULL rows were "
    "backfilled with 'Unknown' by Alembic revision 7b3a1f0c92d4."
)
def test_org_filter_no_org_sentinel(session_factory):
    from orm.logging_functions import get_question_audit_page

    s = session_factory()
    _seed_roles(s)
    _seed_user(s, id=10, username="alice", org="Acme")
    _seed_user(s, id=11, username="bob", org=None)
    now = datetime(2026, 6, 1, 12, 0, 0)
    _add_question(s, user_id=10, content="acme-q", when=now)
    _add_question(s, user_id=11, content="noorg-q", when=now)

    noorg = get_question_audit_page(_filters(orgs=["(no org)"]), page=1, page_size=50)
    assert [it["question"] for it in noorg["items"]] == ["noorg-q"]

    acme = get_question_audit_page(_filters(orgs=["Acme"]), page=1, page_size=50)
    assert [it["question"] for it in acme["items"]] == ["acme-q"]


def test_date_filter_excludes_old_questions(session_factory):
    from orm.logging_functions import get_question_audit_page

    s = session_factory()
    _seed_roles(s)
    _seed_user(s, id=10, username="alice")
    now = datetime.utcnow()
    _add_question(s, user_id=10, content="recent", when=now - timedelta(days=1))
    _add_question(s, user_id=10, content="old", when=now - timedelta(days=60))

    result = get_question_audit_page(_filters(days=7), page=1, page_size=50)
    qs = [it["question"] for it in result["items"]]
    assert "recent" in qs
    assert "old" not in qs


def test_search_matches_question_content(session_factory):
    from orm.logging_functions import get_question_audit_page

    s = session_factory()
    _seed_roles(s)
    _seed_user(s, id=10, username="alice")
    now = datetime(2026, 6, 1, 12, 0, 0)
    _add_question(s, user_id=10, content="What about John Doe today?", when=now)
    _add_question(s, user_id=10, content="Show me yesterday's totals", when=now - timedelta(minutes=1))

    result = get_question_audit_page(_filters(search="john doe"), page=1, page_size=50)
    qs = [it["question"] for it in result["items"]]
    assert qs == ["What about John Doe today?"]


def test_search_matches_assistant_sql_content(session_factory):
    from orm.logging_functions import get_question_audit_page

    s = session_factory()
    _seed_roles(s)
    _seed_user(s, id=10, username="alice")
    now = datetime(2026, 6, 1, 12, 0, 0)
    _add_question(
        s,
        user_id=10,
        content="how many?",
        when=now,
        sql="SELECT count(*) FROM patient WHERE mrn = 12345",
    )
    _add_question(s, user_id=10, content="unrelated", when=now - timedelta(minutes=1))

    result = get_question_audit_page(_filters(search="mrn = 12345"), page=1, page_size=50)
    qs = [it["question"] for it in result["items"]]
    assert qs == ["how many?"]


def test_status_success_summary(session_factory):
    from orm.logging_functions import get_question_audit_page

    s = session_factory()
    _seed_roles(s)
    _seed_user(s, id=10, username="alice")
    now = datetime(2026, 6, 1, 12, 0, 0)
    _add_question(s, user_id=10, content="ok", when=now, summary="here you go")

    result = get_question_audit_page(_filters(), page=1, page_size=50)
    assert result["items"][0]["status"] == "Success"


def test_status_error(session_factory):
    from orm.logging_functions import get_question_audit_page

    s = session_factory()
    _seed_roles(s)
    _seed_user(s, id=10, username="alice")
    now = datetime(2026, 6, 1, 12, 0, 0)
    _add_question(s, user_id=10, content="fail", when=now, summary="partial", error="boom")

    result = get_question_audit_page(_filters(), page=1, page_size=50)
    assert result["items"][0]["status"] == "Error"


def test_status_empty(session_factory):
    from orm.logging_functions import get_question_audit_page

    s = session_factory()
    _seed_roles(s)
    _seed_user(s, id=10, username="alice")
    now = datetime(2026, 6, 1, 12, 0, 0)
    _add_question(s, user_id=10, content="hanging", when=now)

    result = get_question_audit_page(_filters(), page=1, page_size=50)
    assert result["items"][0]["status"] == "Empty"


def test_elapsed_is_sum_across_assistant_rows(session_factory):
    from orm.logging_functions import get_question_audit_page

    s = session_factory()
    _seed_roles(s)
    _seed_user(s, id=10, username="alice")
    now = datetime(2026, 6, 1, 12, 0, 0)
    _add_question(s, user_id=10, content="q", when=now, elapsed=1.5, sql="select 1", summary="ok")
    last_assistant = (
        s.query(Message)
        .filter(
            Message.role == RoleType.ASSISTANT.value, Message.question == "q", Message.type == MessageType.SUMMARY.value
        )
        .first()
    )
    last_assistant.elapsed_time = 0.25
    s.commit()

    result = get_question_audit_page(_filters(), page=1, page_size=50)
    assert result["items"][0]["elapsed_seconds"] == pytest.approx(1.75)


@pytest.mark.skip(
    reason="Epic #179: organization is now NOT NULL on thrive_user, so the legacy "
    "'(no org)' bucket never appears in filter options. Existing NULL rows were "
    "backfilled with 'Unknown' by Alembic revision 7b3a1f0c92d4."
)
def test_filter_options_returns_distinct_sorted(session_factory):
    from orm.logging_functions import get_question_audit_filter_options

    s = session_factory()
    _seed_roles(s)
    _seed_user(s, id=10, username="alice", org="Acme")
    _seed_user(s, id=11, username="bob", org="Beta")
    _seed_user(s, id=12, username="carol", org=None)
    _seed_user(s, id=13, username="dave_no_questions", org="Gamma")  # no questions → excluded
    now = datetime.utcnow()
    _add_question(s, user_id=10, content="q1", when=now - timedelta(days=1))
    _add_question(s, user_id=11, content="q2", when=now - timedelta(days=1))
    _add_question(s, user_id=12, content="q3", when=now - timedelta(days=1))

    opts = get_question_audit_filter_options(days=30)
    assert opts["usernames"] == ["alice", "bob", "carol"]
    assert opts["orgs"] == ["(no org)", "Acme", "Beta"]


def test_export_returns_full_filtered_set_no_pagination(session_factory):
    from orm.logging_functions import get_question_audit_export

    s = session_factory()
    _seed_roles(s)
    _seed_user(s, id=10, username="alice")
    base = datetime(2026, 6, 1, 12, 0, 0)
    for i in range(7):
        _add_question(s, user_id=10, content=f"q{i}", when=base - timedelta(hours=i))

    rows = get_question_audit_export(_filters())
    assert len(rows) == 7
    # Same item shape as page items
    assert "question" in rows[0]
    assert "sql_text" in rows[0]
    assert "status" in rows[0]
