"""Tests for the restored "Top Users by Questions" and new "Top Organizations
by Questions" sections on Admin → Analytics → Overview.

Both sections were cut by Epic #144 / #147 and the per-user one is being
restored alongside a new per-org companion at Joe's request after the
2026-06-15 demo. Each row pairs a horizontal bar chart with a stats grid.
"""

from __future__ import annotations

import datetime as dt

import pytest

from orm.models import Message, RoleTypeEnum, User, UserRole
from utils.enums import MessageType, RoleType


@pytest.fixture
def patched_session(monkeypatch, in_memory_orm_session):
    """Point ``views.admin_analytics.SessionLocal`` at the in-memory DB and
    clear the ``_read_user_org_question_stats`` cache between tests."""
    from views import admin_analytics

    monkeypatch.setattr(admin_analytics, "SessionLocal", in_memory_orm_session)
    admin_analytics._read_user_org_question_stats.clear()
    yield in_memory_orm_session
    admin_analytics._read_user_org_question_stats.clear()


def _mk_user(username, organization, role_id):
    return User(
        username=username,
        first_name=username.title(),
        last_name="Tester",
        password="x",
        email=f"{username}@{organization.lower().replace(' ', '')}.example",
        organization=organization,
        user_role_id=role_id,
    )


def _seed_users(session_factory):
    """Seed three users across two orgs and return their IDs by username."""
    with session_factory() as session:
        admin_role = session.query(UserRole).filter_by(role=RoleTypeEnum.ADMIN).one()
        rob = _mk_user("rob", "ThriveAI", admin_role.id)
        cassie = _mk_user("cassie", "ThriveAI", admin_role.id)
        joe = _mk_user("joe", "Erie County Health", admin_role.id)
        session.add_all([rob, cassie, joe])
        session.commit()
        return {"rob": rob.id, "cassie": cassie.id, "joe": joe.id}


def _add_msg(session, user_id, *, role, msg_type, days_ago=0):
    m = Message(
        user_id=user_id,
        role=role,
        type=msg_type,
        content="x",
    )
    session.add(m)
    session.commit()
    # `created_at` has server_default=NOW; backdate after insert for time-window tests.
    last = session.query(Message).order_by(Message.id.desc()).first()
    last.created_at = dt.datetime.now() - dt.timedelta(days=days_ago)
    session.commit()


class TestReadUserOrgQuestionStats:
    """The cached aggregation helper that powers both Overview rows."""

    def test_user_top_orders_by_questions_desc(self, patched_session):
        from views.admin_analytics import _read_user_org_question_stats

        ids = _seed_users(patched_session)
        with patched_session() as session:
            for _ in range(5):
                _add_msg(session, ids["rob"], role=RoleType.USER, msg_type=MessageType.SQL)
            for _ in range(2):
                _add_msg(session, ids["cassie"], role=RoleType.USER, msg_type=MessageType.SQL)
            _add_msg(session, ids["joe"], role=RoleType.USER, msg_type=MessageType.SQL)
            session.commit()

        user_top, _, _, _ = _read_user_org_question_stats(30)

        assert list(user_top["User"]) == ["rob", "cassie", "joe"]
        assert list(user_top["Questions"]) == [5, 2, 1]

    def test_user_top_excludes_zero_question_users(self, patched_session):
        from views.admin_analytics import _read_user_org_question_stats

        ids = _seed_users(patched_session)
        with patched_session() as session:
            _add_msg(session, ids["rob"], role=RoleType.USER, msg_type=MessageType.SQL)
            session.commit()

        user_top, _, _, _ = _read_user_org_question_stats(30)

        assert list(user_top["User"]) == ["rob"]
        # cassie and joe asked nothing — they don't belong in the chart.
        assert "cassie" not in list(user_top["User"])
        assert "joe" not in list(user_top["User"])

    def test_org_top_aggregates_users_under_same_organization(self, patched_session):
        from views.admin_analytics import _read_user_org_question_stats

        ids = _seed_users(patched_session)
        with patched_session() as session:
            # ThriveAI: rob 5, cassie 2 → 7
            for _ in range(5):
                _add_msg(session, ids["rob"], role=RoleType.USER, msg_type=MessageType.SQL)
            for _ in range(2):
                _add_msg(session, ids["cassie"], role=RoleType.USER, msg_type=MessageType.SQL)
            # Erie County Health: joe 4
            for _ in range(4):
                _add_msg(session, ids["joe"], role=RoleType.USER, msg_type=MessageType.SQL)
            session.commit()

        _, _, org_top, _ = _read_user_org_question_stats(30)

        assert list(org_top["Organization"]) == ["ThriveAI", "Erie County Health"]
        assert list(org_top["Questions"]) == [7, 4]

    def test_time_filter_excludes_questions_older_than_window(self, patched_session):
        from views.admin_analytics import _read_user_org_question_stats

        ids = _seed_users(patched_session)
        with patched_session() as session:
            # In-window
            _add_msg(session, ids["rob"], role=RoleType.USER, msg_type=MessageType.SQL, days_ago=1)
            _add_msg(session, ids["rob"], role=RoleType.USER, msg_type=MessageType.SQL, days_ago=6)
            # Out-of-window for a 7-day query (45 days ago)
            _add_msg(session, ids["rob"], role=RoleType.USER, msg_type=MessageType.SQL, days_ago=45)
            session.commit()

        user_top_7, _, _, _ = _read_user_org_question_stats(7)
        user_top_90, _, _, _ = _read_user_org_question_stats(90)

        assert int(user_top_7.loc[user_top_7["User"] == "rob", "Questions"].iloc[0]) == 2
        assert int(user_top_90.loc[user_top_90["User"] == "rob", "Questions"].iloc[0]) == 3

    def test_stats_grid_breaks_out_dataframes_summaries_charts_errors(self, patched_session):
        from views.admin_analytics import _read_user_org_question_stats

        ids = _seed_users(patched_session)
        with patched_session() as session:
            _add_msg(session, ids["rob"], role=RoleType.USER, msg_type=MessageType.SQL)
            _add_msg(session, ids["rob"], role=RoleType.ASSISTANT, msg_type=MessageType.DATAFRAME)
            _add_msg(session, ids["rob"], role=RoleType.ASSISTANT, msg_type=MessageType.SUMMARY)
            _add_msg(session, ids["rob"], role=RoleType.ASSISTANT, msg_type=MessageType.PLOTLY_CHART)
            _add_msg(session, ids["rob"], role=RoleType.ASSISTANT, msg_type=MessageType.ERROR)
            session.commit()

        _, user_stats, _, _ = _read_user_org_question_stats(30)
        rob = user_stats[user_stats["User"] == "rob"].iloc[0]

        assert rob["Questions"] == 1
        assert rob["DataFrames"] == 1
        assert rob["Summaries"] == 1
        assert rob["Charts"] == 1
        assert rob["Errors"] == 1

    def test_user_stats_includes_users_with_no_activity(self, patched_session):
        """The full-population grid keeps zero-activity users visible so admins
        can see who has never asked anything — the chart caps to 10 actives,
        but the grid is the complete roster.
        """
        from views.admin_analytics import _read_user_org_question_stats

        _seed_users(patched_session)
        _, user_stats, _, _ = _read_user_org_question_stats(30)

        assert set(user_stats["User"]) == {"rob", "cassie", "joe"}
        assert all(int(q) == 0 for q in user_stats["Questions"])

    def test_org_stats_includes_orgs_with_no_activity(self, patched_session):
        from views.admin_analytics import _read_user_org_question_stats

        _seed_users(patched_session)
        _, _, _, org_stats = _read_user_org_question_stats(30)

        assert set(org_stats["Organization"]) == {"ThriveAI", "Erie County Health"}

    def test_user_top_caps_at_ten_rows(self, patched_session):
        from views.admin_analytics import _read_user_org_question_stats

        with patched_session() as session:
            admin_role = session.query(UserRole).filter_by(role=RoleTypeEnum.ADMIN).one()
            for i in range(15):
                u = _mk_user(f"u{i:02d}", "ThriveAI", admin_role.id)
                session.add(u)
                session.flush()
                # Each user asks i+1 questions so order is u14, u13, ..., u00.
                for _ in range(i + 1):
                    _add_msg(session, u.id, role=RoleType.USER, msg_type=MessageType.SQL)
            session.commit()

        user_top, _, _, _ = _read_user_org_question_stats(30)

        assert len(user_top) == 10
        assert user_top["User"].iloc[0] == "u14"
        assert user_top["Questions"].iloc[0] == 15
