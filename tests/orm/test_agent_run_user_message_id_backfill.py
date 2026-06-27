"""Backfill verification for the AgentRun.user_message_id backfill migration.

Companion to the production fix in ``agent/deps_builder.py`` that populates
``AgentRun.user_message_id`` on new runs. Existing rows have NULL there, which
makes the per-query audit view's join
(``AgentRun.user_message_id == Message.id`` in
``orm/logging_functions.py``) miss and every historical row fall through to
the legacy half of the UNION ALL — labelling Pipeline/Scope/Tool as
"legacy" / "Legacy/Unknown" / "(legacy SQL)" even when the run was agentic.

The migration retroactively wires each ``AgentRun`` to the USER ``Message``
row that triggered it by matching ``(user_id, question)`` and picking the
most recent qualifying message at or before the run's ``created_at``.
"""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

_PRIOR_HEAD = "9a4c12e7b001"


def _cfg(tmp_path):
    url = f"sqlite:///{tmp_path / 'thrive.sqlite3'}"
    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", url)
    return url, cfg


def _seed_user(conn, user_id: int = 1):
    conn.execute(
        text(
            "INSERT INTO thrive_user_role (id, role_name, description, role) "
            "VALUES (4, 'Patient', 'Patient access', 'PATIENT')"
        )
    )
    conn.execute(
        text(
            "INSERT INTO thrive_user (id, username, first_name, last_name, "
            "password, email, organization, user_role_id) "
            f"VALUES ({user_id}, 'u{user_id}', 'F', 'L', 'x', "
            f"'u{user_id}@x.com', 'Org', 4)"
        )
    )


def _insert_message(conn, *, mid, user_id, content, created_at, role="user"):
    conn.execute(
        text(
            "INSERT INTO thrive_message (id, user_id, role, content, type, created_at) "
            "VALUES (:id, :uid, :role, :content, 'TEXT', :ts)"
        ),
        {"id": mid, "uid": user_id, "role": role, "content": content, "ts": created_at},
    )


def _insert_run(conn, *, run_id, user_id, question, created_at, user_message_id=None):
    conn.execute(
        text(
            "INSERT INTO thrive_agent_run "
            "(run_id, session_id, user_id, user_role, question, user_message_id, "
            "status, success, tool_call_count, event_count, review_status, "
            "logging_mode, schema_version, created_at) "
            "VALUES (:rid, 's', :uid, 1, :q, :umid, 'complete', 1, 0, 0, "
            "'unreviewed', 'full', 1, :ts)"
        ),
        {
            "rid": run_id,
            "uid": user_id,
            "q": question,
            "umid": user_message_id,
            "ts": created_at,
        },
    )


def _get_run_umid(conn, run_id: str):
    return conn.execute(
        text("SELECT user_message_id FROM thrive_agent_run WHERE run_id = :r"),
        {"r": run_id},
    ).scalar()


def test_backfill_populates_user_message_id_for_matching_run(tmp_path):
    """A USER Message inserted just before an AgentRun with same (user_id,
    question) is matched on backfill."""
    url, cfg = _cfg(tmp_path)
    command.upgrade(cfg, _PRIOR_HEAD)
    engine = create_engine(url)

    with engine.begin() as conn:
        _seed_user(conn)
        _insert_message(
            conn,
            mid=100,
            user_id=1,
            content="How many patients?",
            created_at="2026-06-10 10:00:00",
        )
        _insert_run(
            conn,
            run_id="r-A",
            user_id=1,
            question="How many patients?",
            created_at="2026-06-10 10:00:02",
        )

    command.upgrade(cfg, "head")

    with engine.connect() as conn:
        assert _get_run_umid(conn, "r-A") == 100


def test_backfill_leaves_unrelated_runs_alone(tmp_path):
    """No matching Message (different content) → user_message_id stays NULL."""
    url, cfg = _cfg(tmp_path)
    command.upgrade(cfg, _PRIOR_HEAD)
    engine = create_engine(url)

    with engine.begin() as conn:
        _seed_user(conn)
        _insert_message(
            conn,
            mid=100,
            user_id=1,
            content="Different question",
            created_at="2026-06-10 10:00:00",
        )
        _insert_run(
            conn,
            run_id="r-orphan",
            user_id=1,
            question="Original question",
            created_at="2026-06-10 10:00:02",
        )

    command.upgrade(cfg, "head")

    with engine.connect() as conn:
        assert _get_run_umid(conn, "r-orphan") is None


def test_backfill_preserves_already_populated_user_message_id(tmp_path):
    """Runs whose user_message_id is already set must not be re-assigned."""
    url, cfg = _cfg(tmp_path)
    command.upgrade(cfg, _PRIOR_HEAD)
    engine = create_engine(url)

    with engine.begin() as conn:
        _seed_user(conn)
        _insert_message(
            conn,
            mid=100,
            user_id=1,
            content="Already linked",
            created_at="2026-06-10 10:00:00",
        )
        _insert_message(
            conn,
            mid=101,
            user_id=1,
            content="Already linked",
            created_at="2026-06-10 10:00:01",
        )
        # Run already correctly points at the older Message 100; the newer
        # 101 would win on backfill if we didn't gate on NULL.
        _insert_run(
            conn,
            run_id="r-prelinked",
            user_id=1,
            question="Already linked",
            created_at="2026-06-10 10:00:02",
            user_message_id=100,
        )

    command.upgrade(cfg, "head")

    with engine.connect() as conn:
        assert _get_run_umid(conn, "r-prelinked") == 100


def test_backfill_picks_message_per_run_when_question_repeated(tmp_path):
    """Two AgentRuns with the same question (asked twice) each match their
    own USER Message — the latest one at-or-before each run's created_at."""
    url, cfg = _cfg(tmp_path)
    command.upgrade(cfg, _PRIOR_HEAD)
    engine = create_engine(url)

    with engine.begin() as conn:
        _seed_user(conn)
        _insert_message(
            conn,
            mid=200,
            user_id=1,
            content="Repeated question",
            created_at="2026-06-10 10:00:00",
        )
        _insert_run(
            conn,
            run_id="r-first",
            user_id=1,
            question="Repeated question",
            created_at="2026-06-10 10:00:01",
        )
        _insert_message(
            conn,
            mid=201,
            user_id=1,
            content="Repeated question",
            created_at="2026-06-10 10:00:05",
        )
        _insert_run(
            conn,
            run_id="r-second",
            user_id=1,
            question="Repeated question",
            created_at="2026-06-10 10:00:06",
        )

    command.upgrade(cfg, "head")

    with engine.connect() as conn:
        assert _get_run_umid(conn, "r-first") == 200
        assert _get_run_umid(conn, "r-second") == 201


def test_backfill_ignores_messages_from_other_users(tmp_path):
    """Match must be scoped to AgentRun.user_id — a Message owned by a
    different user with the same content does not satisfy the join."""
    url, cfg = _cfg(tmp_path)
    command.upgrade(cfg, _PRIOR_HEAD)
    engine = create_engine(url)

    with engine.begin() as conn:
        _seed_user(conn, user_id=1)
        conn.execute(
            text(
                "INSERT INTO thrive_user (id, username, first_name, last_name, "
                "password, email, organization, user_role_id) "
                "VALUES (2, 'u2', 'F', 'L', 'x', 'u2@x.com', 'Org', 4)"
            )
        )
        _insert_message(
            conn,
            mid=300,
            user_id=2,
            content="Cross-user question",
            created_at="2026-06-10 10:00:00",
        )
        _insert_run(
            conn,
            run_id="r-u1",
            user_id=1,
            question="Cross-user question",
            created_at="2026-06-10 10:00:02",
        )

    command.upgrade(cfg, "head")

    with engine.connect() as conn:
        assert _get_run_umid(conn, "r-u1") is None


def test_backfill_matches_across_unbounded_lookback(tmp_path):
    """The backfill scans the whole table — a USER Message from days before
    the run still satisfies the match as long as it's the latest qualifying
    one at-or-before the run's created_at."""
    url, cfg = _cfg(tmp_path)
    command.upgrade(cfg, _PRIOR_HEAD)
    engine = create_engine(url)

    with engine.begin() as conn:
        _seed_user(conn)
        _insert_message(
            conn,
            mid=500,
            user_id=1,
            content="Long-gap question",
            created_at="2026-06-08 09:00:00",
        )
        _insert_run(
            conn,
            run_id="r-gap",
            user_id=1,
            question="Long-gap question",
            created_at="2026-06-10 10:00:00",
        )

    command.upgrade(cfg, "head")

    with engine.connect() as conn:
        assert _get_run_umid(conn, "r-gap") == 500


def test_backfill_ignores_assistant_role_messages(tmp_path):
    """Assistant-role rows with the same content (e.g. echoed in a SQL
    Message's question column) must not satisfy the join — only USER rows."""
    url, cfg = _cfg(tmp_path)
    command.upgrade(cfg, _PRIOR_HEAD)
    engine = create_engine(url)

    with engine.begin() as conn:
        _seed_user(conn)
        _insert_message(
            conn,
            mid=400,
            user_id=1,
            content="Role-filtered question",
            created_at="2026-06-10 10:00:00",
            role="assistant",
        )
        _insert_run(
            conn,
            run_id="r-asst",
            user_id=1,
            question="Role-filtered question",
            created_at="2026-06-10 10:00:02",
        )

    command.upgrade(cfg, "head")

    with engine.connect() as conn:
        assert _get_run_umid(conn, "r-asst") is None
