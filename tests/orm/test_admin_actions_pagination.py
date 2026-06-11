"""Tests for ``orm.logging_functions.get_admin_actions_page`` (#156).

Covers:
  - Item shape exposes all 14 AdminAction-derived fields per spec.
  - Page math: page out of range returns empty items + correct total.
  - Page size variants (25 / 50 / 100 / 200).
  - Default ordering is created_at DESC.
  - Time-range (``days``) filtering excludes older rows.
  - Joined admin_username falls back to "Unknown" when the admin row is missing.
  - Error path returns ``{"items": [], "total": 0}``.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from orm.models import AdminAction, Base, RoleTypeEnum, User, UserRole


# Fields the dict-shape contract guarantees per the #156 spec.
EXPECTED_ITEM_KEYS = {
    "id",
    "created_at",
    "admin_username",
    "action_type",
    "target_user_id",
    "target_username",
    "target_entity_type",
    "target_entity_id",
    "description",
    "old_value",
    "new_value",
    "affected_count",
    "success",
    "error_message",
}


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
    session.add(UserRole(id=1, role_name="Admin", description="Admin", role=RoleTypeEnum.ADMIN))
    session.commit()


def _seed_admin(session, *, user_id=1, username="alice_admin"):
    session.add(
        User(
            id=user_id,
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


def _add_action(
    session,
    *,
    admin_id=1,
    action_type="user_update",
    target_user_id=None,
    target_username=None,
    target_entity_type=None,
    target_entity_id=None,
    description="A change",
    old_value=None,
    new_value=None,
    affected_count=None,
    success=True,
    error_message=None,
    when=None,
):
    action = AdminAction(
        admin_id=admin_id,
        action_type=action_type,
        target_user_id=target_user_id,
        target_username=target_username,
        target_entity_type=target_entity_type,
        target_entity_id=target_entity_id,
        description=description,
        old_value=old_value,
        new_value=new_value,
        affected_count=affected_count,
        success=success,
        error_message=error_message,
    )
    session.add(action)
    session.commit()
    if when is not None:
        # server_default=func.now() is applied on insert; override after so
        # the time-range filter can be exercised deterministically.
        last = session.query(AdminAction).order_by(AdminAction.id.desc()).first()
        last.created_at = when
        session.commit()
    return action


# ---------------------------------------------------------------------------
# Shape & contract
# ---------------------------------------------------------------------------


class TestItemShape:
    def test_item_exposes_all_fourteen_fields(self, session_factory):
        """Each item must expose the full set of AdminAction-derived fields
        named in the #156 spec."""
        from orm.logging_functions import get_admin_actions_page

        with session_factory() as s:
            _seed_roles(s)
            _seed_admin(s, user_id=1, username="alice_admin")
            _seed_admin(s, user_id=2, username="bob_target")
            _add_action(
                s,
                admin_id=1,
                action_type="user_update",
                target_user_id=2,
                target_username="bob_target",
                target_entity_type="user",
                target_entity_id="2",
                description="Updated bob",
                old_value=json.dumps({"role": "Patient"}),
                new_value=json.dumps({"role": "Doctor"}),
                affected_count=1,
                success=True,
                error_message=None,
            )

        result = get_admin_actions_page(days=30, page=1, page_size=50)
        assert result["total"] == 1
        assert len(result["items"]) == 1
        item = result["items"][0]
        assert set(item.keys()) == EXPECTED_ITEM_KEYS, (
            f"Item keys mismatch.\nMissing: {EXPECTED_ITEM_KEYS - set(item.keys())}\n"
            f"Extra: {set(item.keys()) - EXPECTED_ITEM_KEYS}"
        )
        # Spot-check joined + projected values.
        assert item["admin_username"] == "alice_admin"
        assert item["target_user_id"] == 2
        assert item["target_username"] == "bob_target"
        assert item["target_entity_type"] == "user"
        assert item["target_entity_id"] == "2"
        assert item["affected_count"] == 1
        assert item["success"] is True
        assert item["error_message"] is None
        assert item["old_value"] is not None and "Patient" in item["old_value"]
        assert item["new_value"] is not None and "Doctor" in item["new_value"]

    def test_missing_admin_falls_back_to_unknown(self, session_factory):
        """If the AdminAction.admin_id row no longer exists (e.g. user
        deleted), admin_username is "Unknown" via the outer join."""
        from orm.logging_functions import get_admin_actions_page

        with session_factory() as s:
            _seed_roles(s)
            # No User seeded — admin_id=999 won't join.
            _add_action(s, admin_id=999, action_type="user_delete")

        result = get_admin_actions_page(days=30)
        assert result["total"] == 1
        assert result["items"][0]["admin_username"] == "Unknown"

    def test_nullable_target_fields_remain_none(self, session_factory):
        """When the action has no target user/entity, the dict still carries
        the keys with None values."""
        from orm.logging_functions import get_admin_actions_page

        with session_factory() as s:
            _seed_roles(s)
            _seed_admin(s)
            _add_action(s, target_user_id=None, target_username=None, target_entity_type=None, target_entity_id=None)

        item = get_admin_actions_page(days=30)["items"][0]
        assert item["target_user_id"] is None
        assert item["target_username"] is None
        assert item["target_entity_type"] is None
        assert item["target_entity_id"] is None


# ---------------------------------------------------------------------------
# Page math
# ---------------------------------------------------------------------------


class TestPagination:
    def test_total_independent_of_page(self, session_factory):
        from orm.logging_functions import get_admin_actions_page

        with session_factory() as s:
            _seed_roles(s)
            _seed_admin(s)
            for i in range(7):
                _add_action(s, description=f"Action {i}")

        assert get_admin_actions_page(days=30, page=1, page_size=3)["total"] == 7
        assert get_admin_actions_page(days=30, page=2, page_size=3)["total"] == 7
        assert get_admin_actions_page(days=30, page=99, page_size=3)["total"] == 7

    def test_page_size_caps_returned_items(self, session_factory):
        from orm.logging_functions import get_admin_actions_page

        with session_factory() as s:
            _seed_roles(s)
            _seed_admin(s)
            for i in range(12):
                _add_action(s, description=f"Action {i}")

        assert len(get_admin_actions_page(days=30, page=1, page_size=5)["items"]) == 5
        assert len(get_admin_actions_page(days=30, page=2, page_size=5)["items"]) == 5
        # Last page has remainder only.
        assert len(get_admin_actions_page(days=30, page=3, page_size=5)["items"]) == 2

    def test_page_out_of_range_returns_empty_items_with_correct_total(self, session_factory):
        from orm.logging_functions import get_admin_actions_page

        with session_factory() as s:
            _seed_roles(s)
            _seed_admin(s)
            for i in range(3):
                _add_action(s, description=f"Action {i}")

        out = get_admin_actions_page(days=30, page=99, page_size=10)
        assert out["items"] == []
        assert out["total"] == 3

    def test_page_size_variants_25_50_100_200(self, session_factory):
        """Spec calls out [25, 50, 100, 200] as the selectbox options; this
        check just confirms each is accepted without error and returns at
        most that many items."""
        from orm.logging_functions import get_admin_actions_page

        with session_factory() as s:
            _seed_roles(s)
            _seed_admin(s)
            for i in range(30):
                _add_action(s, description=f"Action {i}")

        for size in (25, 50, 100, 200):
            out = get_admin_actions_page(days=30, page=1, page_size=size)
            assert out["total"] == 30
            assert len(out["items"]) == min(size, 30)


# ---------------------------------------------------------------------------
# Ordering & time-range filter
# ---------------------------------------------------------------------------


class TestOrderingAndTimeRange:
    def test_default_order_is_created_at_desc(self, session_factory):
        from orm.logging_functions import get_admin_actions_page

        now = datetime.now()
        with session_factory() as s:
            _seed_roles(s)
            _seed_admin(s)
            _add_action(s, description="oldest", when=now - timedelta(hours=3))
            _add_action(s, description="middle", when=now - timedelta(hours=2))
            _add_action(s, description="newest", when=now - timedelta(hours=1))

        items = get_admin_actions_page(days=30)["items"]
        assert [i["description"] for i in items] == ["newest", "middle", "oldest"]

    def test_days_window_excludes_old_rows(self, session_factory):
        from orm.logging_functions import get_admin_actions_page

        now = datetime.now()
        with session_factory() as s:
            _seed_roles(s)
            _seed_admin(s)
            _add_action(s, description="recent", when=now - timedelta(days=2))
            _add_action(s, description="too_old", when=now - timedelta(days=45))

        out = get_admin_actions_page(days=7)
        assert out["total"] == 1
        assert out["items"][0]["description"] == "recent"


# ---------------------------------------------------------------------------
# Error path
# ---------------------------------------------------------------------------


class TestErrorPath:
    def test_db_exception_returns_empty_sentinel(self, monkeypatch):
        """When the SessionLocal raises, get_admin_actions_page must return
        the documented sentinel ``{"items": [], "total": 0}`` and log a warning
        (mirrors get_question_audit_page's error path)."""
        from orm import logging_functions as lf

        def _boom(*_a, **_kw):
            raise RuntimeError("db is on fire")

        monkeypatch.setattr(lf, "SessionLocal", _boom)
        out = lf.get_admin_actions_page(days=30)
        assert out == {"items": [], "total": 0}
