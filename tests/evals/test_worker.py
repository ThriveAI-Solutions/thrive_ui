"""Durable queue claim serialization, heartbeat recovery, and cancellation."""

import asyncio
import json
import threading
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from evals.cases import NormalizedCase, NormalizedTurn
from orm.evaluation_models import EvaluationCase, EvaluationCaseResult, EvaluationRun
from orm.models import Base, RoleTypeEnum, User, UserRole


@pytest.fixture
def file_db(tmp_path, monkeypatch):
    db = tmp_path / "eval.sqlite3"
    engine = create_engine(
        f"sqlite:///{db}",
        connect_args={"check_same_thread": False, "timeout": 30},
    )
    Base.metadata.create_all(engine)
    SL = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    monkeypatch.setattr("orm.evaluation_functions.SessionLocal", SL)
    monkeypatch.setattr("evals.worker.engine", engine)
    yield SimpleNamespace(SL=SL, engine=engine)
    engine.dispose()


def _seed_admin(session):
    ur = UserRole(role_name="Admin", description="ADMIN", role=RoleTypeEnum.ADMIN)
    session.add(ur)
    session.flush()
    u = User(
        username="admin",
        first_name="T",
        last_name="U",
        password="x",
        email="admin@e.test",
        organization="Thrive",
        user_role_id=ur.id,
    )
    session.add(u)
    session.flush()
    return u


def _payload(case_id):
    return json.dumps(
        NormalizedCase(
            case_id=case_id,
            version=1,
            source_type="feedback",
            title="t",
            patient_source_id="src-1",
            patient_label="",
            turns=(NormalizedTurn(role="main", prompt="Q?"),),
        ).to_payload()
    )


def _seed_run(session, admin_id, *, execution_mode="asynchronous", status="queued", n_cases=1, created_offset=0):
    run = EvaluationRun(
        run_id=f"run-{status}-{created_offset}-{execution_mode}",
        run_type="feedback_batch",
        execution_mode=execution_mode,
        status=status,
        requested_by=admin_id,
        total_cases=n_cases,
        completed_cases=0,
        failed_cases=0,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=created_offset),
    )
    session.add(run)
    session.flush()
    for ordinal in range(n_cases):
        case = EvaluationCase(
            case_id=f"{run.run_id}-c{ordinal}",
            version=1,
            source_type="feedback",
            status="active",
            payload_json=_payload(f"{run.run_id}-c{ordinal}"),
            created_by=admin_id,
        )
        session.add(case)
        session.flush()
        session.add(
            EvaluationCaseResult(
                evaluation_run_id=run.id,
                evaluation_case_id=case.id,
                ordinal=ordinal,
                attempt=1,
                status="pending",
            )
        )
    session.flush()
    return run


async def _fake_execute_case(case, resources):
    from evals.executor import CaseExecution

    return CaseExecution(
        case_id=case.case_id,
        status="completed",
        patient={"source_id": case.patient_source_id},
        turns=({"index": 0, "role": "main", "answer": "A", "tool_calls": [], "judge": None},),
    )


class TestClaimSerialization:
    def test_only_one_of_two_claimers_wins(self, file_db):
        from evals.worker import SqliteEvaluationQueue

        with file_db.SL() as s:
            admin = _seed_admin(s)
            _seed_run(s, admin.id, status="queued")
            s.commit()

        results: list = []
        barrier = threading.Barrier(2)
        queue = SqliteEvaluationQueue()

        def claim(worker_id):
            barrier.wait()
            results.append(queue.claim_next(worker_id))

        threads = [threading.Thread(target=claim, args=(f"w{i}",)) for i in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        claimed = [r for r in results if r is not None]
        assert len(claimed) == 1

    def test_second_queued_run_waits_while_one_active(self, file_db):
        from evals.worker import SqliteEvaluationQueue

        with file_db.SL() as s:
            admin = _seed_admin(s)
            _seed_run(s, admin.id, status="queued", created_offset=0)
            _seed_run(s, admin.id, status="queued", created_offset=10)
            s.commit()

        queue = SqliteEvaluationQueue()
        first = queue.claim_next("w1")
        assert first is not None
        # One run now running -> claim must refuse the second.
        second = queue.claim_next("w2")
        assert second is None
        with file_db.SL() as s:
            queued = s.query(EvaluationRun).filter(EvaluationRun.status == "queued").count()
            assert queued == 1

    def test_claim_ignores_synchronous_runs(self, file_db):
        from evals.worker import SqliteEvaluationQueue

        with file_db.SL() as s:
            admin = _seed_admin(s)
            _seed_run(s, admin.id, status="queued", execution_mode="synchronous")
            s.commit()
        assert SqliteEvaluationQueue().claim_next("w1") is None


class TestRecovery:
    def test_stale_running_run_is_requeued(self, file_db):
        from orm.evaluation_functions import mark_interrupted_runs

        now = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
        with file_db.SL() as s:
            admin = _seed_admin(s)
            run = _seed_run(s, admin.id, status="running", n_cases=2)
            run.heartbeat_at = now - timedelta(minutes=5)
            results = (
                s.query(EvaluationCaseResult)
                .filter(EvaluationCaseResult.evaluation_run_id == run.id)
                .order_by(EvaluationCaseResult.ordinal)
                .all()
            )
            results[0].status = "completed"
            results[1].status = "running"
            s.commit()
            run_id = run.run_id

        requeued = mark_interrupted_runs(now, heartbeat_timeout_s=120.0)
        assert requeued == 1
        with file_db.SL() as s:
            run = s.query(EvaluationRun).filter(EvaluationRun.run_id == run_id).one()
            assert run.status == "queued"
            statuses = sorted(
                r.status
                for r in s.query(EvaluationCaseResult).filter(EvaluationCaseResult.evaluation_run_id == run.id)
            )
            # completed case preserved; running case reset to pending for resume.
            assert statuses == ["completed", "pending"]

    def test_fresh_heartbeat_is_not_requeued(self, file_db):
        from orm.evaluation_functions import mark_interrupted_runs

        now = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
        with file_db.SL() as s:
            admin = _seed_admin(s)
            run = _seed_run(s, admin.id, status="running")
            run.heartbeat_at = now - timedelta(seconds=5)
            s.commit()
        assert mark_interrupted_runs(now, heartbeat_timeout_s=120.0) == 0


class TestCancellation:
    def test_cancel_marks_remaining_pending_cancelled(self, file_db):
        from orm.evaluation_functions import run_evaluation_cases

        with file_db.SL() as s:
            admin = _seed_admin(s)
            run = _seed_run(s, admin.id, status="running", n_cases=3)
            run.cancel_requested = True
            results = (
                s.query(EvaluationCaseResult)
                .filter(EvaluationCaseResult.evaluation_run_id == run.id)
                .order_by(EvaluationCaseResult.ordinal)
                .all()
            )
            # First case already finished before the cancel landed.
            results[0].status = "completed"
            s.commit()
            run_id = run.run_id

        view = asyncio.run(run_evaluation_cases(run_id, SimpleNamespace()))
        assert view.status == "cancelled"
        with file_db.SL() as s:
            statuses = sorted(
                r.status
                for r in s.query(EvaluationCaseResult).join(EvaluationRun).filter(EvaluationRun.run_id == run_id)
            )
            assert statuses == ["cancelled", "cancelled", "completed"]


class TestWorkerLifecycle:
    def test_start_is_idempotent(self, file_db, monkeypatch):
        import evals.worker as worker

        monkeypatch.setattr(worker, "_worker_thread", None)
        monkeypatch.setattr(worker, "_stop_event", None)
        # A no-op queue so the loop never does real work.
        idle_queue = SimpleNamespace(
            claim_next=lambda w: None, heartbeat=lambda r, w: None, finish=lambda r, s: None
        )
        monkeypatch.setattr("orm.evaluation_functions.purge_expired_evaluation_payloads", lambda now: 0)
        monkeypatch.setattr("orm.evaluation_functions.mark_interrupted_runs", lambda now, t=120.0: 0)
        try:
            worker.start_evaluation_worker(queue=idle_queue)
            first = worker._worker_thread
            worker.start_evaluation_worker(queue=idle_queue)
            assert worker._worker_thread is first
            assert sum(1 for t in threading.enumerate() if t.name == "evaluation-worker") == 1
        finally:
            worker.stop_evaluation_worker()
