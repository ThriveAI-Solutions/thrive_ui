"""Tests for ``orm.agent_logging_functions.get_patient_audit_autocomplete``
(Epic #190, Phase 3).

Backs the By-Patient audit tab's patient picker. One row per distinct
``(source_id, display_name)`` pair touched within the time window, ordered
by ``last_touched_at DESC``.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from orm.models import AgentPatientAccess, Base, RoleTypeEnum, User, UserRole


@pytest.fixture
def session_factory(monkeypatch):
    from orm import agent_logging_functions as alf

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    monkeypatch.setattr(alf, "SessionLocal", Session)
    return Session


def _seed_user(session, *, id=10, username="alice"):
    session.add(
        UserRole(id=4, role_name="Patient", description="Patient", role=RoleTypeEnum.PATIENT),
    )
    session.commit()
    session.add(
        User(
            id=id,
            user_role_id=4,
            username=username,
            first_name=username,
            last_name="X",
            password="hash:abc",
            email=f"{username}@x.io",
            organization="Acme",
        )
    )
    session.commit()


def _add_access(
    session,
    *,
    source_id,
    display_name=None,
    created_at,
    run_id="run-1",
    user_id=10,
    access_type="read",
    access_origin="tool",
):
    session.add(
        AgentPatientAccess(
            run_id=run_id,
            session_id=f"sess-{run_id}",
            user_id=user_id,
            source_id=source_id,
            display_name=display_name,
            access_type=access_type,
            access_origin=access_origin,
        )
    )
    session.commit()
    row = (
        session.query(AgentPatientAccess)
        .filter_by(source_id=source_id, display_name=display_name)
        .order_by(AgentPatientAccess.id.desc())
        .first()
    )
    row.created_at = created_at
    session.commit()


def test_empty_database_returns_empty_list(session_factory):
    from orm.agent_logging_functions import get_patient_audit_autocomplete

    session_factory()  # creates schema via the fixture's Base.metadata.create_all
    assert get_patient_audit_autocomplete() == []


def test_three_distinct_source_ids_ordered_by_last_touched_desc(session_factory):
    from orm.agent_logging_functions import get_patient_audit_autocomplete

    s = session_factory()
    _seed_user(s)
    base = datetime(2026, 6, 1, 12, 0, 0)
    _add_access(s, source_id="pat-1", display_name="Alice", created_at=base - timedelta(minutes=10))
    _add_access(s, source_id="pat-2", display_name="Bob", created_at=base - timedelta(minutes=5))
    _add_access(s, source_id="pat-3", display_name="Carol", created_at=base)

    rows = get_patient_audit_autocomplete()
    assert [r["source_id"] for r in rows] == ["pat-3", "pat-2", "pat-1"]
    assert all(r["access_count"] == 1 for r in rows)


def test_repeated_touches_aggregate_into_one_row(session_factory):
    from orm.agent_logging_functions import get_patient_audit_autocomplete

    s = session_factory()
    _seed_user(s)
    base = datetime(2026, 6, 1, 12, 0, 0)
    for i in range(4):
        _add_access(s, source_id="pat-1", display_name="Alice", created_at=base - timedelta(minutes=i))

    rows = get_patient_audit_autocomplete()
    assert len(rows) == 1
    assert rows[0]["access_count"] == 4
    # last_touched_at is the most recent (i=0).
    assert rows[0]["last_touched_at"] == base


def test_same_source_id_two_display_names_returns_two_rows(session_factory):
    from orm.agent_logging_functions import get_patient_audit_autocomplete

    s = session_factory()
    _seed_user(s)
    base = datetime(2026, 6, 1, 12, 0, 0)
    _add_access(s, source_id="pat-1", display_name="Alice A", created_at=base - timedelta(minutes=5))
    _add_access(s, source_id="pat-1", display_name="Alice Anders", created_at=base)

    rows = get_patient_audit_autocomplete()
    assert len(rows) == 2
    # Newest first.
    assert rows[0]["display_name"] == "Alice Anders"
    assert rows[1]["display_name"] == "Alice A"


def test_older_than_days_window_is_excluded(session_factory):
    from orm.agent_logging_functions import get_patient_audit_autocomplete

    s = session_factory()
    _seed_user(s)
    now = datetime.now()
    _add_access(s, source_id="recent", display_name="R", created_at=now - timedelta(days=1))
    _add_access(s, source_id="ancient", display_name="A", created_at=now - timedelta(days=90))

    rows = get_patient_audit_autocomplete(days=7)
    assert {r["source_id"] for r in rows} == {"recent"}


def test_query_filter_matches_source_id_substring(session_factory):
    from orm.agent_logging_functions import get_patient_audit_autocomplete

    s = session_factory()
    _seed_user(s)
    base = datetime(2026, 6, 1, 12, 0, 0)
    _add_access(s, source_id="pat-12-abc", display_name="X", created_at=base)
    _add_access(s, source_id="pat-99-xyz", display_name="Y", created_at=base - timedelta(minutes=1))

    rows = get_patient_audit_autocomplete(query="pat-12")
    assert [r["source_id"] for r in rows] == ["pat-12-abc"]


def test_query_filter_matches_display_name_substring(session_factory):
    from orm.agent_logging_functions import get_patient_audit_autocomplete

    s = session_factory()
    _seed_user(s)
    base = datetime(2026, 6, 1, 12, 0, 0)
    _add_access(s, source_id="pat-A", display_name="Alice Anders", created_at=base)
    _add_access(s, source_id="pat-B", display_name="Bob Brown", created_at=base - timedelta(minutes=1))

    rows = get_patient_audit_autocomplete(query="alice")
    assert [r["source_id"] for r in rows] == ["pat-A"]
