"""Upgrade/downgrade verification for the authenticated evaluation workspace migration.

Covers every table, index, unique/check constraint, and foreign key added by
``alembic/versions/d4f27a91c8b3_add_authenticated_evaluations.py`` on top of
the ``orm/evaluation_models.py`` declarations (AgentRunFeedback,
AgentRunFeedbackEvent, EvaluationCase, EvaluationRun, EvaluationCaseResult,
EvaluationReviewEvent, AdminNotification).
"""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

_PRIOR_HEAD = "c233fb500001"

_NEW_TABLES = [
    "thrive_agent_run_feedback",
    "thrive_agent_run_feedback_event",
    "thrive_evaluation_case",
    "thrive_evaluation_run",
    "thrive_evaluation_case_result",
    "thrive_evaluation_review_event",
    "thrive_admin_notification",
]

_EXPECTED_INDEXES = {
    "thrive_agent_run_feedback": {
        "ix_thrive_agent_run_feedback_agent_run": ("agent_run_id",),
        "ix_thrive_agent_run_feedback_user": ("user_id",),
        "ix_thrive_agent_run_feedback_rating": ("rating",),
        "ix_thrive_agent_run_feedback_created": ("created_at",),
    },
    "thrive_agent_run_feedback_event": {
        "ix_thrive_agent_run_feedback_event_feedback": ("feedback_id",),
        "ix_thrive_agent_run_feedback_event_actor": ("actor_user_id",),
        "ix_thrive_agent_run_feedback_event_created": ("created_at",),
    },
    "thrive_evaluation_case": {
        "ix_thrive_evaluation_case_case_id": ("case_id",),
        "ix_thrive_evaluation_case_source_status": ("source_type", "status"),
        "ix_thrive_evaluation_case_source_feedback": ("source_feedback_id",),
        "ix_thrive_evaluation_case_source_message": ("source_message_id",),
        "ix_thrive_evaluation_case_source_agent_run": ("source_agent_run_id",),
        "ix_thrive_evaluation_case_promoted_from": ("promoted_from_case_id",),
        "ix_thrive_evaluation_case_created_by": ("created_by",),
    },
    "thrive_evaluation_run": {
        "ix_thrive_evaluation_run_run_id": ("run_id",),
        "ix_thrive_evaluation_run_requested_by": ("requested_by",),
        "ix_thrive_evaluation_run_status_time": ("status", "created_at"),
        "ix_thrive_evaluation_run_heartbeat": ("heartbeat_at",),
    },
    "thrive_evaluation_case_result": {
        "ix_thrive_evaluation_case_result_run_ordinal": ("evaluation_run_id", "ordinal"),
        "ix_thrive_evaluation_case_result_case": ("evaluation_case_id",),
        "ix_thrive_evaluation_case_result_status": ("status",),
        "ix_thrive_evaluation_case_result_reviewer": ("reviewed_by",),
    },
    "thrive_evaluation_review_event": {
        "ix_thrive_evaluation_review_event_result": ("result_id",),
        "ix_thrive_evaluation_review_event_reviewer": ("reviewer_id",),
        "ix_thrive_evaluation_review_event_created": ("created_at",),
    },
    "thrive_admin_notification": {
        "ix_thrive_admin_notification_user": ("user_id",),
        "ix_thrive_admin_notification_run": ("evaluation_run_id",),
        "ix_thrive_admin_notification_unread": ("user_id", "read_at", "created_at"),
    },
}

_EXPECTED_FKS = {
    "thrive_agent_run_feedback": {
        ("agent_run_id",): ("thrive_agent_run", "CASCADE"),
        ("user_id",): ("thrive_user", "CASCADE"),
    },
    "thrive_agent_run_feedback_event": {
        ("feedback_id",): ("thrive_agent_run_feedback", "CASCADE"),
        ("actor_user_id",): ("thrive_user", None),
    },
    "thrive_evaluation_case": {
        ("source_feedback_id",): ("thrive_agent_run_feedback", None),
        ("source_message_id",): ("thrive_message", None),
        ("source_agent_run_id",): ("thrive_agent_run", None),
        ("promoted_from_case_id",): ("thrive_evaluation_case", None),
        ("created_by",): ("thrive_user", None),
    },
    "thrive_evaluation_run": {
        ("requested_by",): ("thrive_user", None),
    },
    "thrive_evaluation_case_result": {
        ("evaluation_run_id",): ("thrive_evaluation_run", "CASCADE"),
        ("evaluation_case_id",): ("thrive_evaluation_case", None),
        ("reviewed_by",): ("thrive_user", None),
    },
    "thrive_evaluation_review_event": {
        ("result_id",): ("thrive_evaluation_case_result", "CASCADE"),
        ("reviewer_id",): ("thrive_user", None),
    },
    "thrive_admin_notification": {
        ("user_id",): ("thrive_user", "CASCADE"),
        ("evaluation_run_id",): ("thrive_evaluation_run", "CASCADE"),
    },
}


def _cfg(tmp_path):
    url = f"sqlite:///{tmp_path / 'thrive.sqlite3'}"
    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", url)
    return url, cfg


def _upgrade_to_head(tmp_path):
    url, cfg = _cfg(tmp_path)
    command.upgrade(cfg, "head")
    return create_engine(url), cfg


def _seed_base(conn):
    """Seed the minimum pre-existing rows FK-dependent inserts need."""
    conn.execute(
        text(
            "INSERT INTO thrive_user_role (id, role_name, description, role) "
            "VALUES (4, 'Patient', 'Patient access', 'PATIENT')"
        )
    )
    for uid in (1, 2):
        conn.execute(
            text(
                "INSERT INTO thrive_user (id, username, first_name, last_name, "
                "password, email, organization, user_role_id) "
                f"VALUES ({uid}, 'u{uid}', 'F', 'L', 'x', 'u{uid}@x.com', 'Org', 4)"
            )
        )
    conn.execute(
        text(
            "INSERT INTO thrive_message (id, user_id, role, content, type, created_at) "
            "VALUES (100, 1, 'user', 'How many patients?', 'TEXT', '2026-07-01 00:00:00')"
        )
    )
    for rid, run_id in ((1, "run-1"), (2, "run-2")):
        conn.execute(
            text(
                "INSERT INTO thrive_agent_run "
                "(id, run_id, session_id, user_id, user_role, question, "
                "status, success, tool_call_count, event_count, review_status, "
                "logging_mode, schema_version, created_at) "
                "VALUES (:id, :rid, 's', 1, 1, 'Q', 'complete', 1, 0, 0, "
                "'unreviewed', 'full', 1, '2026-07-01 00:00:01')"
            ),
            {"id": rid, "rid": run_id},
        )


def _insert_feedback(conn, *, agent_run_id=1, user_id=1, rating="up"):
    return conn.execute(
        text(
            "INSERT INTO thrive_agent_run_feedback (agent_run_id, user_id, rating) "
            "VALUES (:arid, :uid, :rating)"
        ),
        {"arid": agent_run_id, "uid": user_id, "rating": rating},
    ).lastrowid


def _insert_evaluation_run(conn, *, run_id="eval-1", requested_by=1):
    return conn.execute(
        text(
            "INSERT INTO thrive_evaluation_run "
            "(run_id, run_type, execution_mode, status, requested_by, total_cases, "
            "completed_cases, failed_cases, cancel_requested) "
            "VALUES (:rid, 'feedback_batch', 'async', 'queued', :req, 1, 0, 0, 0)"
        ),
        {"rid": run_id, "req": requested_by},
    ).lastrowid


def _insert_evaluation_case(conn, *, case_id="case-1", version=1, source_type="curated", created_by=1):
    return conn.execute(
        text(
            "INSERT INTO thrive_evaluation_case "
            "(case_id, version, source_type, status, payload_json, created_by) "
            "VALUES (:cid, :ver, :stype, 'active', '{}', :cb)"
        ),
        {"cid": case_id, "ver": version, "stype": source_type, "cb": created_by},
    ).lastrowid


def _insert_case_result(conn, *, evaluation_run_id, evaluation_case_id, ordinal=1, attempt=1):
    return conn.execute(
        text(
            "INSERT INTO thrive_evaluation_case_result "
            "(evaluation_run_id, evaluation_case_id, ordinal, attempt, status) "
            "VALUES (:run, :case, :ord, :att, 'pending')"
        ),
        {"run": evaluation_run_id, "case": evaluation_case_id, "ord": ordinal, "att": attempt},
    ).lastrowid


# --------------------------------------------------------------------------
# Structural: tables, indexes, foreign keys
# --------------------------------------------------------------------------


def test_upgrade_creates_all_new_tables(tmp_path):
    engine, _ = _upgrade_to_head(tmp_path)
    table_names = set(inspect(engine).get_table_names())
    for table in _NEW_TABLES:
        assert table in table_names


def test_upgrade_creates_expected_indexes(tmp_path):
    engine, _ = _upgrade_to_head(tmp_path)
    inspector = inspect(engine)
    for table, expected in _EXPECTED_INDEXES.items():
        actual = {ix["name"]: tuple(ix["column_names"]) for ix in inspector.get_indexes(table)}
        for name, columns in expected.items():
            assert name in actual, f"missing index {name} on {table}"
            assert actual[name] == columns, f"index {name} columns {actual[name]} != {columns}"
    # run_id's explicit index must be unique per the model's Index(..., unique=True)
    # SQLite reflection reports this as a plain int (from PRAGMA index_list), not
    # the bool singleton, so compare truthiness rather than identity.
    run_indexes = {ix["name"]: ix for ix in inspector.get_indexes("thrive_evaluation_run")}
    assert bool(run_indexes["ix_thrive_evaluation_run_run_id"]["unique"]) is True


def test_upgrade_creates_expected_foreign_keys(tmp_path):
    engine, _ = _upgrade_to_head(tmp_path)
    inspector = inspect(engine)
    for table, expected in _EXPECTED_FKS.items():
        actual = {
            tuple(fk["constrained_columns"]): (fk["referred_table"], fk["options"].get("ondelete"))
            for fk in inspector.get_foreign_keys(table)
        }
        for columns, (referred_table, ondelete) in expected.items():
            assert columns in actual, f"missing FK {columns} on {table}"
            assert actual[columns] == (referred_table, ondelete), (
                f"FK {columns} on {table} = {actual[columns]}, expected ({referred_table}, {ondelete})"
            )


# --------------------------------------------------------------------------
# Functional: constraint enforcement and cascade behavior
# --------------------------------------------------------------------------


def test_agent_run_feedback_unique_owner_and_rating_check(tmp_path):
    engine, _ = _upgrade_to_head(tmp_path)
    with engine.begin() as conn:
        _seed_base(conn)
        _insert_feedback(conn, agent_run_id=1, user_id=1, rating="up")

    with pytest.raises(IntegrityError):
        with engine.begin() as conn:
            _insert_feedback(conn, agent_run_id=1, user_id=1, rating="down")

    with pytest.raises(IntegrityError):
        with engine.begin() as conn:
            _insert_feedback(conn, agent_run_id=2, user_id=1, rating="sideways")


def test_agent_run_feedback_cascades_on_agent_run_and_user_delete(tmp_path):
    engine, _ = _upgrade_to_head(tmp_path)
    with engine.begin() as conn:
        _seed_base(conn)
        _insert_feedback(conn, agent_run_id=1, user_id=1, rating="up")

    with engine.begin() as conn:
        conn.execute(text("PRAGMA foreign_keys=ON"))
        conn.execute(text("DELETE FROM thrive_agent_run WHERE id = 1"))
        remaining = conn.execute(text("SELECT COUNT(*) FROM thrive_agent_run_feedback")).scalar()
        assert remaining == 0


def test_agent_run_feedback_event_cascades_on_feedback_delete(tmp_path):
    engine, _ = _upgrade_to_head(tmp_path)
    with engine.begin() as conn:
        _seed_base(conn)
        feedback_id = _insert_feedback(conn, agent_run_id=1, user_id=1, rating="up")
        conn.execute(
            text(
                "INSERT INTO thrive_agent_run_feedback_event "
                "(feedback_id, actor_user_id, old_rating, new_rating) "
                "VALUES (:fid, 1, NULL, 'up')"
            ),
            {"fid": feedback_id},
        )

    with engine.begin() as conn:
        conn.execute(text("PRAGMA foreign_keys=ON"))
        conn.execute(text("DELETE FROM thrive_agent_run_feedback WHERE id = :fid"), {"fid": feedback_id})
        remaining = conn.execute(text("SELECT COUNT(*) FROM thrive_agent_run_feedback_event")).scalar()
        assert remaining == 0


def test_evaluation_case_unique_version_check_and_self_fk(tmp_path):
    engine, _ = _upgrade_to_head(tmp_path)
    with engine.begin() as conn:
        _seed_base(conn)
        first_id = _insert_evaluation_case(conn, case_id="case-1", version=1, source_type="curated")

    with pytest.raises(IntegrityError):
        with engine.begin() as conn:
            _insert_evaluation_case(conn, case_id="case-1", version=1, source_type="feedback")

    with pytest.raises(IntegrityError):
        with engine.begin() as conn:
            _insert_evaluation_case(conn, case_id="case-2", version=1, source_type="bogus")

    with engine.begin() as conn:
        promoted_id = conn.execute(
            text(
                "INSERT INTO thrive_evaluation_case "
                "(case_id, version, source_type, status, promoted_from_case_id, payload_json, created_by) "
                "VALUES ('case-1', 2, 'curated', 'active', :parent, '{}', 1)"
            ),
            {"parent": first_id},
        ).lastrowid
        parent = conn.execute(
            text("SELECT promoted_from_case_id FROM thrive_evaluation_case WHERE id = :id"),
            {"id": promoted_id},
        ).scalar()
        assert parent == first_id


def test_evaluation_run_run_id_unique(tmp_path):
    engine, _ = _upgrade_to_head(tmp_path)
    with engine.begin() as conn:
        _seed_base(conn)
        _insert_evaluation_run(conn, run_id="eval-dup", requested_by=1)

    with pytest.raises(IntegrityError):
        with engine.begin() as conn:
            _insert_evaluation_run(conn, run_id="eval-dup", requested_by=1)


def test_evaluation_case_result_unique_attempt_and_restrict_on_case_delete(tmp_path):
    engine, _ = _upgrade_to_head(tmp_path)
    with engine.begin() as conn:
        _seed_base(conn)
        run_id = _insert_evaluation_run(conn, run_id="eval-2", requested_by=1)
        case_id = _insert_evaluation_case(conn, case_id="case-3", version=1, source_type="curated")
        _insert_case_result(conn, evaluation_run_id=run_id, evaluation_case_id=case_id, ordinal=1, attempt=1)

    with pytest.raises(IntegrityError):
        with engine.begin() as conn:
            _insert_case_result(conn, evaluation_run_id=run_id, evaluation_case_id=case_id, ordinal=2, attempt=1)

    with pytest.raises(IntegrityError):
        with engine.begin() as conn:
            conn.execute(text("PRAGMA foreign_keys=ON"))
            conn.execute(text("DELETE FROM thrive_evaluation_case WHERE id = :id"), {"id": case_id})


def test_evaluation_case_result_cascades_on_evaluation_run_delete(tmp_path):
    engine, _ = _upgrade_to_head(tmp_path)
    with engine.begin() as conn:
        _seed_base(conn)
        run_id = _insert_evaluation_run(conn, run_id="eval-3", requested_by=1)
        case_id = _insert_evaluation_case(conn, case_id="case-4", version=1, source_type="curated")
        _insert_case_result(conn, evaluation_run_id=run_id, evaluation_case_id=case_id)

    with engine.begin() as conn:
        conn.execute(text("PRAGMA foreign_keys=ON"))
        conn.execute(text("DELETE FROM thrive_evaluation_run WHERE id = :id"), {"id": run_id})
        remaining = conn.execute(text("SELECT COUNT(*) FROM thrive_evaluation_case_result")).scalar()
        assert remaining == 0


def test_evaluation_review_event_verdict_check_and_cascade(tmp_path):
    engine, _ = _upgrade_to_head(tmp_path)
    with engine.begin() as conn:
        _seed_base(conn)
        run_id = _insert_evaluation_run(conn, run_id="eval-4", requested_by=1)
        case_id = _insert_evaluation_case(conn, case_id="case-5", version=1, source_type="curated")
        result_id = _insert_case_result(conn, evaluation_run_id=run_id, evaluation_case_id=case_id)

    with pytest.raises(IntegrityError):
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO thrive_evaluation_review_event (result_id, reviewer_id, new_verdict) "
                    "VALUES (:rid, 1, 'maybe')"
                ),
                {"rid": result_id},
            )

    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO thrive_evaluation_review_event (result_id, reviewer_id, new_verdict) "
                "VALUES (:rid, 1, 'correct')"
            ),
            {"rid": result_id},
        )

    with engine.begin() as conn:
        conn.execute(text("PRAGMA foreign_keys=ON"))
        conn.execute(text("DELETE FROM thrive_evaluation_case_result WHERE id = :id"), {"id": result_id})
        remaining = conn.execute(text("SELECT COUNT(*) FROM thrive_evaluation_review_event")).scalar()
        assert remaining == 0


def test_admin_notification_cascades_on_evaluation_run_delete(tmp_path):
    engine, _ = _upgrade_to_head(tmp_path)
    with engine.begin() as conn:
        _seed_base(conn)
        run_id = _insert_evaluation_run(conn, run_id="eval-5", requested_by=1)
        conn.execute(
            text(
                "INSERT INTO thrive_admin_notification (user_id, evaluation_run_id, kind) "
                "VALUES (1, :run, 'run_complete')"
            ),
            {"run": run_id},
        )

    with engine.begin() as conn:
        conn.execute(text("PRAGMA foreign_keys=ON"))
        conn.execute(text("DELETE FROM thrive_evaluation_run WHERE id = :id"), {"id": run_id})
        remaining = conn.execute(text("SELECT COUNT(*) FROM thrive_admin_notification")).scalar()
        assert remaining == 0


# --------------------------------------------------------------------------
# Downgrade
# --------------------------------------------------------------------------


def test_downgrade_removes_all_new_tables(tmp_path):
    url, cfg = _cfg(tmp_path)
    command.upgrade(cfg, "head")
    engine = create_engine(url)
    table_names = set(inspect(engine).get_table_names())
    for table in _NEW_TABLES:
        assert table in table_names
    engine.dispose()

    command.downgrade(cfg, _PRIOR_HEAD)
    engine = create_engine(url)
    table_names = set(inspect(engine).get_table_names())
    for table in _NEW_TABLES:
        assert table not in table_names
    # Pre-existing tables from the parent revision must be untouched.
    assert "thrive_agent_run" in table_names
    assert "thrive_user" in table_names
