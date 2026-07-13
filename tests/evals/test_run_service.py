"""Durable evaluation-run creation, synchronous execution, and admin verdicts."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from evals.cases import NormalizedCase, NormalizedTurn
from orm.evaluation_models import (
    AdminNotification,
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


def _make_user(session, *, role, username):
    ur = UserRole(role_name=f"{role.name}-{username}", description=role.name, role=role)
    session.add(ur)
    session.flush()
    u = User(
        username=username,
        first_name="T",
        last_name="U",
        password="x",
        email=f"{username}@e.test",
        organization="Thrive",
        user_role_id=ur.id,
    )
    session.add(u)
    session.flush()
    return u


def _payload(case_id, patient="src-1"):
    return json.dumps(
        NormalizedCase(
            case_id=case_id,
            version=1,
            source_type="feedback",
            title="t",
            patient_source_id=patient,
            patient_label="",
            turns=(NormalizedTurn(role="main", prompt="Q?"),),
        ).to_payload()
    )


def _make_case(session, *, case_id, source_type="feedback", status="active", created_by):
    case = EvaluationCase(
        case_id=case_id,
        version=1,
        source_type=source_type,
        status=status,
        payload_json=_payload(case_id),
        created_by=created_by,
    )
    session.add(case)
    session.flush()
    return case


@pytest.fixture
def seeded(eval_db):
    with eval_db() as session:
        admin = _make_user(session, role=RoleTypeEnum.ADMIN, username="admin")
        doctor = _make_user(session, role=RoleTypeEnum.DOCTOR, username="doc")
        session.commit()
        return SimpleNamespace(SL=eval_db, admin_id=admin.id, doctor_id=doctor.id)


def _fake_resources(fail_ids=()):
    return SimpleNamespace(fail_ids=set(fail_ids))


async def _fake_execute_case(case, resources):
    from evals.executor import CaseExecution

    if case.case_id in getattr(resources, "fail_ids", set()):
        return CaseExecution(
            case_id=case.case_id,
            status="failed",
            patient={"source_id": case.patient_source_id},
            turns=(),
            error_type="Boom",
            error_message="failed",
        )
    return CaseExecution(
        case_id=case.case_id,
        status="completed",
        patient={"source_id": case.patient_source_id},
        turns=(
            {
                "index": 0,
                "role": "main",
                "answer": "A",
                "tool_calls": [],
                "judge": {"suggestion": "looks_correct", "reason": "matches"},
            },
        ),
    )


class TestCreateRun:
    def test_single_feedback_is_synchronous(self, seeded):
        from orm.evaluation_functions import create_evaluation_run

        with seeded.SL() as s:
            c = _make_case(s, case_id="c1", created_by=seeded.admin_id)
            s.commit()
            cid = c.id
        view = create_evaluation_run([cid], seeded.admin_id)
        assert view.run_type == "single_feedback"
        assert view.execution_mode == "synchronous"
        assert view.total_cases == 1

    def test_two_feedback_is_async_batch(self, seeded):
        from orm.evaluation_functions import create_evaluation_run

        with seeded.SL() as s:
            ids = [
                _make_case(s, case_id="c1", created_by=seeded.admin_id).id,
                _make_case(s, case_id="c2", created_by=seeded.admin_id).id,
            ]
            s.commit()
        view = create_evaluation_run(ids, seeded.admin_id)
        assert view.run_type == "feedback_batch"
        assert view.execution_mode == "asynchronous"

    def test_single_curated_is_subset_async(self, seeded):
        from orm.evaluation_functions import create_evaluation_run

        with seeded.SL() as s:
            # Two active curated cases exist; selecting one is a subset.
            cid = _make_case(s, case_id="cur1", source_type="curated", created_by=seeded.admin_id).id
            _make_case(s, case_id="cur2", source_type="curated", created_by=seeded.admin_id)
            s.commit()
        view = create_evaluation_run([cid], seeded.admin_id)
        assert view.run_type == "curated_subset"
        assert view.execution_mode == "asynchronous"

    def test_all_active_curated_is_full_suite(self, seeded):
        from orm.evaluation_functions import create_evaluation_run

        with seeded.SL() as s:
            ids = [
                _make_case(s, case_id="cur1", source_type="curated", created_by=seeded.admin_id).id,
                _make_case(s, case_id="cur2", source_type="curated", created_by=seeded.admin_id).id,
            ]
            # A non-active curated case must not count against the full suite.
            _make_case(s, case_id="cur3", source_type="curated", status="draft", created_by=seeded.admin_id)
            s.commit()
        view = create_evaluation_run(ids, seeded.admin_id)
        assert view.run_type == "full_suite"

    def test_non_admin_raises_before_insert(self, seeded):
        from orm.evaluation_functions import create_evaluation_run

        with seeded.SL() as s:
            cid = _make_case(s, case_id="c1", created_by=seeded.admin_id).id
            s.commit()
        with pytest.raises(PermissionError):
            create_evaluation_run([cid], seeded.doctor_id)
        with seeded.SL() as s:
            assert s.query(EvaluationRun).count() == 0

    def test_results_inserted_atomically_with_ordinals(self, seeded):
        from orm.evaluation_functions import create_evaluation_run

        with seeded.SL() as s:
            ids = [
                _make_case(s, case_id="c1", created_by=seeded.admin_id).id,
                _make_case(s, case_id="c2", created_by=seeded.admin_id).id,
            ]
            s.commit()
        create_evaluation_run(ids, seeded.admin_id)
        with seeded.SL() as s:
            results = s.query(EvaluationCaseResult).order_by(EvaluationCaseResult.ordinal).all()
            assert [r.ordinal for r in results] == [0, 1]
            assert all(r.status == "pending" for r in results)

    def test_empty_selection_raises(self, seeded):
        from orm.evaluation_functions import EvaluationServiceError, create_evaluation_run

        with pytest.raises(EvaluationServiceError):
            create_evaluation_run([], seeded.admin_id)


class TestSynchronousExecution:
    def test_single_feedback_run_completes(self, seeded, monkeypatch):
        from orm.evaluation_functions import create_evaluation_run, execute_synchronous_run

        monkeypatch.setattr("evals.executor.execute_case", _fake_execute_case)
        with seeded.SL() as s:
            cid = _make_case(s, case_id="c1", created_by=seeded.admin_id).id
            s.commit()
        run = create_evaluation_run([cid], seeded.admin_id)
        view = execute_synchronous_run(run.run_id, seeded.admin_id, resources=_fake_resources())
        assert view.status == "completed"
        assert view.completed_cases == 1
        with seeded.SL() as s:
            result = s.query(EvaluationCaseResult).one()
            assert result.status == "completed"
            assert result.judge_verdict == "looks_correct"
            assert s.query(AdminNotification).count() == 1

    def test_non_admin_cannot_execute(self, seeded, monkeypatch):
        from orm.evaluation_functions import create_evaluation_run, execute_synchronous_run

        monkeypatch.setattr("evals.executor.execute_case", _fake_execute_case)
        with seeded.SL() as s:
            cid = _make_case(s, case_id="c1", created_by=seeded.admin_id).id
            s.commit()
        run = create_evaluation_run([cid], seeded.admin_id)
        with pytest.raises(PermissionError):
            execute_synchronous_run(run.run_id, seeded.doctor_id, resources=_fake_resources())

    def test_mixed_outcomes_aggregate_to_completed_with_errors(self, seeded, monkeypatch):
        from orm.evaluation_functions import create_evaluation_run, run_evaluation_cases

        monkeypatch.setattr("evals.executor.execute_case", _fake_execute_case)
        with seeded.SL() as s:
            ids = [
                _make_case(s, case_id="ok", created_by=seeded.admin_id).id,
                _make_case(s, case_id="bad", created_by=seeded.admin_id).id,
            ]
            s.commit()
        run = create_evaluation_run(ids, seeded.admin_id)
        view = asyncio.run(run_evaluation_cases(run.run_id, _fake_resources(fail_ids={"bad"})))
        assert view.status == "completed_with_errors"
        assert view.completed_cases == 1
        assert view.failed_cases == 1


class TestReview:
    def test_final_verdict_appends_event(self, seeded, monkeypatch):
        from orm.evaluation_functions import (
            create_evaluation_run,
            execute_synchronous_run,
            record_final_verdict,
        )

        monkeypatch.setattr("evals.executor.execute_case", _fake_execute_case)
        with seeded.SL() as s:
            cid = _make_case(s, case_id="c1", created_by=seeded.admin_id).id
            s.commit()
        run = create_evaluation_run([cid], seeded.admin_id)
        execute_synchronous_run(run.run_id, seeded.admin_id, resources=_fake_resources())
        with seeded.SL() as s:
            result_id = s.query(EvaluationCaseResult).one().id

        record_final_verdict(result_id, seeded.admin_id, "incorrect", note="wrong date")
        record_final_verdict(result_id, seeded.admin_id, "cant_tell", note=None)
        with seeded.SL() as s:
            result = s.query(EvaluationCaseResult).one()
            assert result.final_verdict == "cant_tell"
            events = s.query(EvaluationReviewEvent).order_by(EvaluationReviewEvent.id).all()
            assert [e.new_verdict for e in events] == ["incorrect", "cant_tell"]
            assert events[1].old_verdict == "incorrect"

    def test_judge_suggestion_is_not_a_valid_final_verdict(self, seeded):
        from orm.evaluation_functions import EvaluationServiceError, record_final_verdict

        with pytest.raises(EvaluationServiceError):
            record_final_verdict(1, seeded.admin_id, "looks_correct")

    def test_non_admin_cannot_record_verdict(self, seeded):
        from orm.evaluation_functions import record_final_verdict

        with pytest.raises(PermissionError):
            record_final_verdict(1, seeded.doctor_id, "correct")
