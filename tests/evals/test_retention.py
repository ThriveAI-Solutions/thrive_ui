"""Retention purge of expired PHI payloads, keeping audit metadata."""

import json
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from orm.evaluation_models import (
    EvaluationCase,
    EvaluationCaseResult,
    EvaluationReviewEvent,
    EvaluationRun,
)
from orm.models import Base, RoleTypeEnum, User, UserRole


@pytest.fixture
def eval_db(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SL = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    monkeypatch.setattr("orm.evaluation_functions.SessionLocal", SL)
    yield SL
    engine.dispose()


def _admin(session):
    ur = UserRole(role_name="Admin", description="ADMIN", role=RoleTypeEnum.ADMIN)
    session.add(ur)
    session.flush()
    u = User(
        username="admin",
        first_name="T",
        last_name="U",
        password="x",
        email="a@e.test",
        organization="Thrive",
        user_role_id=ur.id,
    )
    session.add(u)
    session.flush()
    return u


def test_expired_case_payloads_purged_audit_kept(eval_db):
    from orm.evaluation_functions import purge_expired_evaluation_payloads

    now = datetime(2026, 6, 1, tzinfo=timezone.utc)
    with eval_db() as s:
        admin = _admin(s)
        expired = EvaluationCase(
            case_id="expired",
            version=1,
            source_type="feedback",
            status="active",
            payload_json=json.dumps({"question": "PHI question", "patient_source_id": "src-1"}),
            created_by=admin.id,
            expires_at=now - timedelta(days=1),
        )
        fresh = EvaluationCase(
            case_id="fresh",
            version=1,
            source_type="feedback",
            status="active",
            payload_json=json.dumps({"question": "still valid"}),
            created_by=admin.id,
            expires_at=now + timedelta(days=30),
        )
        never = EvaluationCase(
            case_id="never",
            version=1,
            source_type="curated",
            status="active",
            payload_json=json.dumps({"question": "no expiry"}),
            created_by=admin.id,
            expires_at=None,
        )
        s.add_all([expired, fresh, never])
        s.flush()
        run = EvaluationRun(
            run_id="run-1",
            run_type="single_feedback",
            execution_mode="synchronous",
            status="completed",
            requested_by=admin.id,
            total_cases=1,
            completed_cases=1,
            failed_cases=0,
        )
        s.add(run)
        s.flush()
        result = EvaluationCaseResult(
            evaluation_run_id=run.id,
            evaluation_case_id=expired.id,
            ordinal=0,
            attempt=1,
            status="completed",
            result_json=json.dumps({"answer": "PHI answer", "turns": []}),
            final_verdict="incorrect",
        )
        s.add(result)
        s.flush()
        s.add(
            EvaluationReviewEvent(result_id=result.id, reviewer_id=admin.id, old_verdict=None, new_verdict="incorrect")
        )
        s.commit()
        expired_id, fresh_id, never_id, result_id, run_id = (
            expired.id,
            fresh.id,
            never.id,
            result.id,
            run.id,
        )

    purged = purge_expired_evaluation_payloads(now)
    assert purged == 1

    with eval_db() as s:
        expired = s.query(EvaluationCase).filter(EvaluationCase.id == expired_id).one()
        assert expired.status == "expired"
        assert "PHI question" not in expired.payload_json

        fresh = s.query(EvaluationCase).filter(EvaluationCase.id == fresh_id).one()
        assert fresh.status == "active"
        assert "still valid" in fresh.payload_json

        never = s.query(EvaluationCase).filter(EvaluationCase.id == never_id).one()
        assert never.status == "active"

        result = s.query(EvaluationCaseResult).filter(EvaluationCaseResult.id == result_id).one()
        assert result.result_json is None
        # Audit metadata (run identity, counts, verdict, review event) survives.
        assert result.final_verdict == "incorrect"
        run = s.query(EvaluationRun).filter(EvaluationRun.id == run_id).one()
        assert run.completed_cases == 1
        assert s.query(EvaluationReviewEvent).count() == 1


def test_idempotent_second_purge_is_noop(eval_db):
    from orm.evaluation_functions import purge_expired_evaluation_payloads

    now = datetime(2026, 6, 1, tzinfo=timezone.utc)
    with eval_db() as s:
        admin = _admin(s)
        s.add(
            EvaluationCase(
                case_id="expired",
                version=1,
                source_type="feedback",
                status="active",
                payload_json=json.dumps({"question": "PHI"}),
                created_by=admin.id,
                expires_at=now - timedelta(days=1),
            )
        )
        s.commit()
    assert purge_expired_evaluation_payloads(now) == 1
    assert purge_expired_evaluation_payloads(now) == 0
