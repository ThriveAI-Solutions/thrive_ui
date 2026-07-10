"""Authenticated evaluation report authorization, shape, and expiry."""

import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from evals.cases import NormalizedCase, NormalizedTurn
from orm.evaluation_models import EvaluationCase, EvaluationCaseResult
from orm.models import Base, RoleTypeEnum, User, UserRole


@pytest.fixture
def eval_db(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SL = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    monkeypatch.setattr("orm.evaluation_functions.SessionLocal", SL)
    yield SL
    engine.dispose()


def _user(session, *, role, username):
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


def _feedback_case(session, admin_id, case_id="c1"):
    payload = NormalizedCase(
        case_id=case_id,
        version=1,
        source_type="feedback",
        title="t",
        patient_source_id="src-1",
        patient_label="Jane",
        turns=(NormalizedTurn(role="main", prompt="Q?"),),
        original_answer="orig",
        reviewer_guidance="[incorrect_answer] wrong",
    ).to_payload()
    case = EvaluationCase(
        case_id=case_id,
        version=1,
        source_type="feedback",
        status="active",
        payload_json=json.dumps(payload),
        created_by=admin_id,
    )
    session.add(case)
    session.flush()
    return case


async def _fake_execute_case(case, resources):
    from evals.executor import CaseExecution

    return CaseExecution(
        case_id=case.case_id,
        status="completed",
        patient={"source_id": case.patient_source_id},
        turns=(
            {
                "index": 0,
                "role": "main",
                "answer": "rerun answer",
                "tool_calls": [],
                "judge": {"suggestion": "looks_wrong", "reason": "mismatch"},
            },
        ),
    )


def _seed_and_run(eval_db, monkeypatch):
    monkeypatch.setattr("evals.executor.execute_case", _fake_execute_case)
    from orm.evaluation_functions import create_evaluation_run, execute_synchronous_run

    with eval_db() as s:
        admin = _user(s, role=RoleTypeEnum.ADMIN, username="admin")
        doctor = _user(s, role=RoleTypeEnum.DOCTOR, username="doc")
        case = _feedback_case(s, admin.id)
        s.commit()
        admin_id, doc_id, case_id = admin.id, doctor.id, case.id
    run = create_evaluation_run([case_id], admin_id)
    execute_synchronous_run(run.run_id, admin_id, resources=object())
    return run.run_id, admin_id, doc_id


def test_report_rejects_non_admin(eval_db, monkeypatch):
    from orm.evaluation_functions import get_evaluation_run_report

    run_id, _admin_id, doc_id = _seed_and_run(eval_db, monkeypatch)
    with pytest.raises(PermissionError):
        get_evaluation_run_report(run_id, doc_id)


def test_report_returns_original_and_rerun_for_admin(eval_db, monkeypatch):
    from orm.evaluation_functions import get_evaluation_run_report

    run_id, admin_id, _doc_id = _seed_and_run(eval_db, monkeypatch)
    report = get_evaluation_run_report(run_id, admin_id)
    assert report["status"] == "completed"
    assert len(report["cases"]) == 1
    case = report["cases"][0]
    assert case["concern"] == "[incorrect_answer] wrong"
    assert case["original_answer"] == "orig"
    assert case["rerun_turns"][0]["answer"] == "rerun answer"
    # LLM triage present but final verdict still unset (admin must decide).
    assert case["judge_verdict"] == "looks_wrong"
    assert case["final_verdict"] is None


def test_report_hides_expired_case_payload(eval_db, monkeypatch):
    from orm.evaluation_functions import get_evaluation_run_report

    run_id, admin_id, _doc_id = _seed_and_run(eval_db, monkeypatch)
    # Simulate retention having purged the case.
    with eval_db() as s:
        case = s.query(EvaluationCase).one()
        case.status = "expired"
        case.payload_json = json.dumps({"expired": True})
        result = s.query(EvaluationCaseResult).one()
        result.result_json = None
        s.commit()

    report = get_evaluation_run_report(run_id, admin_id)
    case = report["cases"][0]
    assert case["expired"] is True
    assert case["concern"] == ""
    assert case["original_answer"] is None
    assert case["rerun_turns"] == []


def test_final_verdict_flows_into_report(eval_db, monkeypatch):
    from orm.evaluation_functions import get_evaluation_run_report, record_final_verdict

    run_id, admin_id, _doc_id = _seed_and_run(eval_db, monkeypatch)
    result_id = get_evaluation_run_report(run_id, admin_id)["cases"][0]["result_id"]
    record_final_verdict(result_id, admin_id, "incorrect", note="date wrong")
    report = get_evaluation_run_report(run_id, admin_id)
    assert report["cases"][0]["final_verdict"] == "incorrect"
    assert report["cases"][0]["review_note"] == "date wrong"
