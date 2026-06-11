"""Tests for orm.logging_functions.get_user_activity_page (#157).

Covers:
  - Page math (page out of range returns empty items + correct total).
  - Page-size variants.
  - Item shape includes all 10 UserActivity-derived fields.
  - Error path returns {"items": [], "total": 0}.
  - Reverse-chronological ordering.
  - Null user_id (failed-login) rows are returned with user_id=None.
  - Days filter excludes rows older than the cutoff.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from orm.models import Base, RoleTypeEnum, User, UserActivity, UserRole


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
    session.add(
        UserRole(
            id=1,
            role_name="Admin",
            description="Admin",
            role=RoleTypeEnum.ADMIN,
        )
    )
    session.commit()


def _seed_user(session, *, id, username):
    session.add(
        User(
            id=id,
            user_role_id=1,
            username=username,
            first_name=username,
            last_name="X",
            password="hash:abc",
            email=f"{username}@x.io",
            organization="TestOrg",  # required per Epic #179
        )
    )
    session.commit()


def _add_activity(
    session,
    *,
    user_id,
    username,
    activity_type,
    when,
    description=None,
    old_value=None,
    new_value=None,
    ip_address=None,
    user_agent=None,
):
    session.add(
        UserActivity(
            user_id=user_id,
            username=username,
            activity_type=activity_type,
            description=description,
            old_value=old_value,
            new_value=new_value,
            ip_address=ip_address,
            user_agent=user_agent,
        )
    )
    session.commit()
    # Override created_at after insert (server_default=func.now()).
    last = session.query(UserActivity).order_by(UserActivity.id.desc()).first()
    last.created_at = when
    session.commit()


def test_empty_database_returns_empty(session_factory):
    from orm.logging_functions import get_user_activity_page

    s = session_factory()
    _seed_roles(s)

    page = get_user_activity_page(days=30, page=1, page_size=50)
    assert page == {"items": [], "total": 0}


def test_returns_all_ten_fields_per_item(session_factory):
    """Each item must expose all 10 UserActivity-derived fields including
    ``user_id`` and ``user_agent`` (which the retired ``get_recent_activity``
    dropped)."""
    from orm.logging_functions import get_user_activity_page

    s = session_factory()
    _seed_roles(s)
    _seed_user(s, id=10, username="alice")
    now = datetime(2026, 6, 1, 12, 0, 0)
    _add_activity(
        s,
        user_id=10,
        username="alice",
        activity_type="login",
        description="User logged in",
        when=now,
        ip_address="127.0.0.1",
        user_agent="Mozilla/5.0 Test",
        old_value='{"a": 1}',
        new_value='{"a": 2}',
    )

    result = get_user_activity_page(days=30, page=1, page_size=50)
    assert result["total"] == 1
    assert len(result["items"]) == 1
    item = result["items"][0]
    expected_keys = {
        "id",
        "created_at",
        "user_id",
        "username",
        "activity_type",
        "description",
        "old_value",
        "new_value",
        "ip_address",
        "user_agent",
    }
    assert set(item.keys()) == expected_keys
    assert item["user_id"] == 10
    assert item["username"] == "alice"
    assert item["activity_type"] == "login"
    assert item["description"] == "User logged in"
    assert item["ip_address"] == "127.0.0.1"
    assert item["user_agent"] == "Mozilla/5.0 Test"
    assert item["old_value"] == '{"a": 1}'
    assert item["new_value"] == '{"a": 2}'
    assert isinstance(item["id"], int)


def test_returns_reverse_chronological_order(session_factory):
    from orm.logging_functions import get_user_activity_page

    s = session_factory()
    _seed_roles(s)
    _seed_user(s, id=10, username="alice")
    now = datetime(2026, 6, 1, 12, 0, 0)
    _add_activity(s, user_id=10, username="alice", activity_type="login", when=now - timedelta(hours=2))
    _add_activity(s, user_id=10, username="alice", activity_type="setting", when=now - timedelta(hours=1))
    _add_activity(s, user_id=10, username="alice", activity_type="logout", when=now)

    result = get_user_activity_page(days=30, page=1, page_size=50)
    types = [it["activity_type"] for it in result["items"]]
    assert types == ["logout", "setting", "login"]
    assert result["total"] == 3


def test_pagination_page_one_returns_first_page(session_factory):
    from orm.logging_functions import get_user_activity_page

    s = session_factory()
    _seed_roles(s)
    _seed_user(s, id=10, username="alice")
    base = datetime(2026, 6, 1, 12, 0, 0)
    for i in range(5):
        _add_activity(
            s,
            user_id=10,
            username="alice",
            activity_type=f"type_{i}",
            when=base - timedelta(hours=i),
        )

    p1 = get_user_activity_page(days=30, page=1, page_size=2)
    assert len(p1["items"]) == 2
    assert p1["total"] == 5


def test_pagination_page_two_returns_next_page(session_factory):
    from orm.logging_functions import get_user_activity_page

    s = session_factory()
    _seed_roles(s)
    _seed_user(s, id=10, username="alice")
    base = datetime(2026, 6, 1, 12, 0, 0)
    for i in range(5):
        _add_activity(
            s,
            user_id=10,
            username="alice",
            activity_type=f"type_{i}",
            when=base - timedelta(hours=i),
        )

    p2 = get_user_activity_page(days=30, page=2, page_size=2)
    assert len(p2["items"]) == 2
    assert p2["total"] == 5
    # Reverse chronological: type_0 newest, type_4 oldest. Page 2 (size 2) is type_2, type_3.
    assert [it["activity_type"] for it in p2["items"]] == ["type_2", "type_3"]


def test_pagination_out_of_range_returns_empty_items_with_correct_total(session_factory):
    """Page beyond the end returns empty items but the total count must still
    be accurate so the UI can render 'Page X of N' correctly."""
    from orm.logging_functions import get_user_activity_page

    s = session_factory()
    _seed_roles(s)
    _seed_user(s, id=10, username="alice")
    base = datetime(2026, 6, 1, 12, 0, 0)
    for i in range(3):
        _add_activity(
            s,
            user_id=10,
            username="alice",
            activity_type=f"t{i}",
            when=base - timedelta(hours=i),
        )

    result = get_user_activity_page(days=30, page=10, page_size=50)
    assert result["items"] == []
    assert result["total"] == 3


@pytest.mark.parametrize("page_size", [25, 50, 100, 200])
def test_pagination_page_size_variants(session_factory, page_size):
    from orm.logging_functions import get_user_activity_page

    s = session_factory()
    _seed_roles(s)
    _seed_user(s, id=10, username="alice")
    base = datetime(2026, 6, 1, 12, 0, 0)
    # Seed 30 rows so 25 is a partial page and 50/100/200 are not.
    for i in range(30):
        _add_activity(
            s,
            user_id=10,
            username="alice",
            activity_type=f"t{i}",
            when=base - timedelta(minutes=i),
        )

    result = get_user_activity_page(days=30, page=1, page_size=page_size)
    assert result["total"] == 30
    assert len(result["items"]) == min(page_size, 30)


def test_null_user_id_failed_login_row_is_returned(session_factory):
    """Failed logins where user lookup failed have null ``user_id``. The page
    helper must return them with ``user_id=None`` so the UI can gate the
    deep-link button accordingly (#157 spec)."""
    from orm.logging_functions import get_user_activity_page

    s = session_factory()
    _seed_roles(s)
    # No user seeded — failed login with null user_id but a captured username.
    now = datetime(2026, 6, 1, 12, 0, 0)
    _add_activity(
        s,
        user_id=None,
        username="ghost",
        activity_type="failed_login",
        when=now,
        ip_address="10.0.0.1",
        user_agent="curl/8.0",
    )

    result = get_user_activity_page(days=30, page=1, page_size=50)
    assert result["total"] == 1
    assert result["items"][0]["user_id"] is None
    assert result["items"][0]["username"] == "ghost"
    assert result["items"][0]["activity_type"] == "failed_login"


def test_days_filter_excludes_old_rows(session_factory):
    """Rows older than ``days`` are excluded from both items and total."""
    from orm.logging_functions import get_user_activity_page

    s = session_factory()
    _seed_roles(s)
    _seed_user(s, id=10, username="alice")
    now = datetime.now()
    _add_activity(s, user_id=10, username="alice", activity_type="recent", when=now - timedelta(days=1))
    _add_activity(s, user_id=10, username="alice", activity_type="ancient", when=now - timedelta(days=400))

    result = get_user_activity_page(days=7, page=1, page_size=50)
    assert result["total"] == 1
    assert result["items"][0]["activity_type"] == "recent"


def test_error_path_returns_empty_envelope(monkeypatch):
    """If the SQLAlchemy session raises, the helper logs a warning and returns
    ``{"items": [], "total": 0}`` so the UI degrades gracefully."""
    from orm import logging_functions as lf

    class _BoomSession:
        def __enter__(self):
            raise RuntimeError("db is on fire")

        def __exit__(self, *_):
            return False

    monkeypatch.setattr(lf, "SessionLocal", lambda: _BoomSession())

    result = lf.get_user_activity_page(days=7, page=1, page_size=50)
    assert result == {"items": [], "total": 0}


def test_get_recent_activity_is_removed():
    """The retired ``get_recent_activity`` helper must no longer exist on the
    module — sole caller was ``_render_activity_tab`` and the swap is complete."""
    from orm import logging_functions as lf

    assert not hasattr(lf, "get_recent_activity"), (
        "get_recent_activity should be removed after the swap to get_user_activity_page (#157)"
    )
