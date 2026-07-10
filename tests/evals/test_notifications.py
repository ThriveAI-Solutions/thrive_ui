"""PHI-free admin notifications for completed evaluation runs."""

import dataclasses
import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from evals.cases import NormalizedCase, NormalizedTurn
from orm.evaluation_models import EvaluationCase
from orm.models import Base, RoleTypeEnum, User, UserRole


@pytest.fixture
def eval_db(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SL = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    monkeypatch.setattr("orm.evaluation_functions.SessionLocal", SL)
    yield SL
    engine.dispose()


def _admin(session, username="admin", role=RoleTypeEnum.ADMIN):
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


def _feedback_case(session, admin_id):
    payload = NormalizedCase(
        case_id="c1",
        version=1,
        source_type="feedback",
        title="PHI TITLE Jane Doe diabetes",
        patient_source_id="src-PHI",
        patient_label="Jane Doe",
        turns=(NormalizedTurn(role="main", prompt="Does Jane Doe have diabetes?"),),
        original_answer="Yes, A1C 7.2",
        reviewer_guidance="[incorrect_answer] wrong date",
    ).to_payload()
    case = EvaluationCase(
        case_id="c1",
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
        turns=({"index": 0, "role": "main", "answer": "A", "tool_calls": [], "judge": None},),
    )


def test_notifications_are_phi_free(eval_db, monkeypatch):
    from orm.evaluation_functions import (
        create_evaluation_run,
        execute_synchronous_run,
        list_admin_notifications,
    )

    monkeypatch.setattr("evals.executor.execute_case", _fake_execute_case)
    with eval_db() as s:
        admin = _admin(s)
        case = _feedback_case(s, admin.id)
        s.commit()
        admin_id, case_id = admin.id, case.id

    run = create_evaluation_run([case_id], admin_id)
    execute_synchronous_run(run.run_id, admin_id, resources=object())

    notes = list_admin_notifications(admin_id)
    assert len(notes) == 1
    note = notes[0]
    assert note.run_id == run.run_id
    assert note.text == f"Evaluation {run.run_id} completed: 1/1 cases."

    # No PHI leaks through the serialized notification view.
    blob = json.dumps(dataclasses.asdict(note), default=str)
    for phi in ("Jane Doe", "diabetes", "src-PHI", "A1C", "wrong date"):
        assert phi not in blob


def test_mark_read_hides_from_unread(eval_db, monkeypatch):
    from orm.evaluation_functions import (
        create_evaluation_run,
        execute_synchronous_run,
        list_admin_notifications,
        mark_notification_read,
    )

    monkeypatch.setattr("evals.executor.execute_case", _fake_execute_case)
    with eval_db() as s:
        admin = _admin(s)
        case = _feedback_case(s, admin.id)
        s.commit()
        admin_id, case_id = admin.id, case.id

    run = create_evaluation_run([case_id], admin_id)
    execute_synchronous_run(run.run_id, admin_id, resources=object())
    note_id = list_admin_notifications(admin_id)[0].notification_id

    mark_notification_read(note_id, admin_id)
    assert list_admin_notifications(admin_id, unread_only=True) == []
    assert len(list_admin_notifications(admin_id, unread_only=False)) == 1


def test_non_admin_cannot_list_notifications(eval_db):
    from orm.evaluation_functions import list_admin_notifications

    with eval_db() as s:
        doctor = _admin(s, username="doc", role=RoleTypeEnum.DOCTOR)
        s.commit()
        doc_id = doctor.id
    with pytest.raises(PermissionError):
        list_admin_notifications(doc_id)
