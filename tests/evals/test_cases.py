"""Curated + feedback case normalization, snapshot immutability, and
execution-mode selection."""

import json
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from evals.cases import (
    CuratedCaseDraft,
    NormalizedCase,
    NormalizedTurn,
    SnapshotUnavailable,
    execution_mode,
    normalize_curated_conversation,
    promote_feedback_case,
    snapshot_feedback_case,
)
from evals.matrix import PlannedConversation, PlannedTurn
from orm.evaluation_models import AgentRunFeedback, EvaluationCase
from orm.models import AgentRun, AgentRunEvent, Base, RoleTypeEnum, User, UserRole

_REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def session_factory(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    testing_session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    monkeypatch.setattr("evals.cases.SessionLocal", testing_session_local)
    yield testing_session_local
    engine.dispose()


def _make_user(session, *, role: RoleTypeEnum, username: str) -> User:
    user_role = UserRole(role_name=f"{role.name.title()}-{username}", description=role.name, role=role)
    session.add(user_role)
    session.flush()
    user = User(
        username=username,
        first_name="Test",
        last_name="User",
        password="x",
        email=f"{username}@example.test",
        organization="Thrive",
        user_role_id=user_role.id,
    )
    session.add(user)
    session.flush()
    return user


def _make_run(session, *, user, logging_mode="full", question="Does this patient have diabetes?"):
    run = AgentRun(
        run_id=f"run-{user.id}-{logging_mode}",
        session_id="session-1",
        group_id="group-1",
        user_id=user.id,
        user_role=1,
        status="success",
        success=True,
        logging_mode=logging_mode,
        question=question,
        selected_patient_source_id="src-123",
        selected_patient_display_name="Jane Doe",
        final_answer_text="Yes, A1C 7.2 on 2026-03-15.",
        message_history_json=json.dumps([{"role": "user", "content": question}]),
    )
    session.add(run)
    session.flush()
    return run


def _make_feedback(session, *, run, user, rating="down", category="incorrect_answer", comment="Wrong date"):
    feedback = AgentRunFeedback(
        agent_run_id=run.id, user_id=user.id, rating=rating, category=category, comment=comment
    )
    session.add(feedback)
    session.commit()
    return feedback


class TestSnapshotFeedbackCase:
    def test_captures_full_fidelity_snapshot(self, session_factory):
        with session_factory() as session:
            admin = _make_user(session, role=RoleTypeEnum.ADMIN, username="admin1")
            owner = _make_user(session, role=RoleTypeEnum.DOCTOR, username="doc1")
            run = _make_run(session, user=owner)
            session.add(
                AgentRunEvent(
                    run_id=run.run_id,
                    seq=1,
                    event_type="tool_call_completed",
                    tool_name="get_patient_clinical_data",
                    payload_summary="1 row; data_availability=data_present",
                    payload_json=json.dumps({"result": {"rows": 1}, "sql_executed": []}),
                )
            )
            feedback = _make_feedback(session, run=run, user=owner)
            session.commit()
            feedback_id, admin_id = feedback.id, admin.id

        case = snapshot_feedback_case(feedback_id, admin_id)
        assert case.source_type == "feedback"
        assert case.status == "active"
        payload = json.loads(case.payload_json)
        assert payload["turns"][0]["prompt"] == "Does this patient have diabetes?"
        assert payload["original_answer"] == "Yes, A1C 7.2 on 2026-03-15."
        assert payload["patient_source_id"] == "src-123"
        assert payload["message_history"]
        assert payload["original_evidence"][0]["result"] == {"rows": 1}
        assert "incorrect_answer" in payload["reviewer_guidance"]
        assert "Wrong date" in payload["reviewer_guidance"]

    def test_summary_string_message_history_is_dropped_not_exploded(self, session_factory):
        # The runtime logs message_history as a summary string
        # ("N prior messages", agent/runner.py), not a list. tuple() over that
        # string would explode it into single characters and crash every replay
        # with "'str' object has no attribute 'conversation_id'". The snapshot
        # must drop it to an empty history instead.
        with session_factory() as session:
            admin = _make_user(session, role=RoleTypeEnum.ADMIN, username="admin1")
            owner = _make_user(session, role=RoleTypeEnum.DOCTOR, username="doc1")
            run = _make_run(session, user=owner)
            run.message_history_json = json.dumps("5 prior messages")
            session.flush()
            feedback = _make_feedback(session, run=run, user=owner)
            session.commit()
            feedback_id, admin_id = feedback.id, admin.id

        case = snapshot_feedback_case(feedback_id, admin_id)
        payload = json.loads(case.payload_json)
        assert payload["message_history"] == []

    def test_immutable_after_feedback_changes(self, session_factory):
        with session_factory() as session:
            admin = _make_user(session, role=RoleTypeEnum.ADMIN, username="admin2")
            owner = _make_user(session, role=RoleTypeEnum.DOCTOR, username="doc2")
            run = _make_run(session, user=owner)
            feedback = _make_feedback(session, run=run, user=owner)
            session.commit()
            feedback_id, admin_id = feedback.id, admin.id

        case = snapshot_feedback_case(feedback_id, admin_id)
        original_payload = case.payload_json

        with session_factory() as session:
            live_feedback = session.query(AgentRunFeedback).filter_by(id=feedback_id).one()
            live_feedback.comment = "Completely different complaint now"
            live_feedback.category = "wrong_data_or_tool"
            session.commit()

        with session_factory() as session:
            stored = session.query(EvaluationCase).filter_by(id=case.id).one()
            assert stored.payload_json == original_payload

    def test_scrubbed_mode_excludes_full_result_rows(self, session_factory):
        with session_factory() as session:
            admin = _make_user(session, role=RoleTypeEnum.ADMIN, username="admin3")
            owner = _make_user(session, role=RoleTypeEnum.DOCTOR, username="doc3")
            run = _make_run(session, user=owner, logging_mode="scrubbed")
            session.add(
                AgentRunEvent(
                    run_id=run.run_id,
                    seq=1,
                    event_type="tool_call_completed",
                    tool_name="get_patient_clinical_data",
                    payload_summary="1 row; data_availability=data_present",
                    payload_json=None,
                )
            )
            feedback = _make_feedback(session, run=run, user=owner)
            session.commit()
            feedback_id, admin_id = feedback.id, admin.id

        case = snapshot_feedback_case(feedback_id, admin_id)
        payload = json.loads(case.payload_json)
        assert payload["original_evidence"][0]["result_summary"] == "1 row; data_availability=data_present"
        assert "result" not in payload["original_evidence"][0]

    def test_disabled_mode_raises_snapshot_unavailable(self, session_factory):
        with session_factory() as session:
            admin = _make_user(session, role=RoleTypeEnum.ADMIN, username="admin4")
            owner = _make_user(session, role=RoleTypeEnum.DOCTOR, username="doc4")
            run = _make_run(session, user=owner, logging_mode="disabled")
            feedback = _make_feedback(session, run=run, user=owner)
            session.commit()
            feedback_id, admin_id = feedback.id, admin.id

        with pytest.raises(SnapshotUnavailable, match="disabled"):
            snapshot_feedback_case(feedback_id, admin_id)

    def test_requires_admin_role(self, session_factory):
        with session_factory() as session:
            owner = _make_user(session, role=RoleTypeEnum.DOCTOR, username="doc5")
            run = _make_run(session, user=owner)
            feedback = _make_feedback(session, run=run, user=owner)
            session.commit()
            feedback_id, owner_id = feedback.id, owner.id

        with pytest.raises(PermissionError):
            snapshot_feedback_case(feedback_id, owner_id)

    def test_only_thumbs_down_feedback_can_be_snapshotted(self, session_factory):
        with session_factory() as session:
            admin = _make_user(session, role=RoleTypeEnum.ADMIN, username="admin6")
            owner = _make_user(session, role=RoleTypeEnum.DOCTOR, username="doc6")
            run = _make_run(session, user=owner)
            feedback = _make_feedback(session, run=run, user=owner, rating="up", category=None, comment=None)
            session.commit()
            feedback_id, admin_id = feedback.id, admin.id

        with pytest.raises(SnapshotUnavailable):
            snapshot_feedback_case(feedback_id, admin_id)

    def test_missing_final_answer_raises(self, session_factory):
        with session_factory() as session:
            admin = _make_user(session, role=RoleTypeEnum.ADMIN, username="admin7")
            owner = _make_user(session, role=RoleTypeEnum.DOCTOR, username="doc7")
            run = _make_run(session, user=owner)
            run.final_answer_text = None
            feedback = _make_feedback(session, run=run, user=owner)
            session.commit()
            feedback_id, admin_id = feedback.id, admin.id

        with pytest.raises(SnapshotUnavailable):
            snapshot_feedback_case(feedback_id, admin_id)


class TestPromoteFeedbackCase:
    def test_creates_draft_case_without_touching_questions_yaml(self, session_factory):
        questions_path = _REPO / "evals/questions.yaml"
        original_mtime = questions_path.stat().st_mtime

        with session_factory() as session:
            admin = _make_user(session, role=RoleTypeEnum.ADMIN, username="admin8")
            source = EvaluationCase(
                case_id="feedback-case-x",
                version=1,
                source_type="feedback",
                status="active",
                payload_json=json.dumps({"schema_version": 1}),
                created_by=admin.id,
            )
            session.add(source)
            session.commit()
            admin_id = admin.id

        draft = promote_feedback_case(
            "feedback-case-x",
            admin_id,
            CuratedCaseDraft(
                prompt="Does this patient have a history of {disease}?",
                followups=("When was this last mentioned?",),
                patient_requirements="Any patient with a chronic condition noted in the problem list",
                expected_behavior="Cites the specific note and date",
                reviewer_guidance="Generalized from a thumbs-down about missing dates",
            ),
        )
        assert draft.source_type == "curated"
        assert draft.status == "draft"
        payload = json.loads(draft.payload_json)
        assert payload["turns"][0]["prompt"] == "Does this patient have a history of {disease}?"
        assert payload["turns"][1]["role"] == "followup"
        assert questions_path.stat().st_mtime == original_mtime

    def test_requires_nonempty_prompt(self, session_factory):
        with session_factory() as session:
            admin = _make_user(session, role=RoleTypeEnum.ADMIN, username="admin9")
            session.commit()
            admin_id = admin.id

        with pytest.raises(ValueError, match="prompt"):
            promote_feedback_case(
                "whatever",
                admin_id,
                CuratedCaseDraft(
                    prompt="",
                    followups=(),
                    patient_requirements="x",
                    expected_behavior="y",
                    reviewer_guidance="z",
                ),
            )

    def test_unknown_case_id_raises(self, session_factory):
        with session_factory() as session:
            admin = _make_user(session, role=RoleTypeEnum.ADMIN, username="admin10")
            session.commit()
            admin_id = admin.id

        with pytest.raises(SnapshotUnavailable):
            promote_feedback_case(
                "does-not-exist",
                admin_id,
                CuratedCaseDraft(
                    prompt="p",
                    followups=(),
                    patient_requirements="x",
                    expected_behavior="y",
                    reviewer_guidance="z",
                ),
            )


def _feedback_case(case_id: str) -> NormalizedCase:
    return NormalizedCase(
        case_id=case_id,
        version=1,
        source_type="feedback",
        title="t",
        patient_source_id="p1",
        patient_label="",
        turns=(NormalizedTurn(role="main", prompt="Q?"),),
    )


def _curated_case(case_id: str) -> NormalizedCase:
    return NormalizedCase(
        case_id=case_id,
        version=1,
        source_type="curated",
        title="t",
        patient_source_id="p1",
        patient_label="",
        turns=(NormalizedTurn(role="main", prompt="Q?"),),
    )


def test_execution_mode_single_feedback_is_synchronous():
    assert execution_mode([_feedback_case("f1")]) == "synchronous"


def test_execution_mode_multiple_feedback_is_asynchronous():
    assert execution_mode([_feedback_case("f1"), _feedback_case("f2")]) == "asynchronous"


def test_execution_mode_curated_is_always_asynchronous():
    assert execution_mode([_curated_case("c1")]) == "asynchronous"


def test_normalize_curated_conversation_wraps_planned_conversation():
    planned = PlannedConversation(
        conversation_id="Q1__src-a",
        question_id="Q1",
        question_title="Invasive procedures in date range",
        reviewer_note="Check date precision",
        source_id="src-a",
        patient_label="patient A",
        turns=[
            PlannedTurn(role="main", prompt="Has this patient had surgery?"),
            PlannedTurn(role="followup", prompt="When?"),
        ],
    )
    normalized = normalize_curated_conversation(planned)
    assert normalized.case_id == "Q1__src-a"
    assert normalized.source_type == "curated"
    assert normalized.patient_source_id == "src-a"
    assert [t.role for t in normalized.turns] == ["main", "followup"]
    assert normalized.reviewer_guidance == "Check date precision"


def test_normalized_case_payload_round_trip():
    case = NormalizedCase(
        case_id="c1",
        version=1,
        source_type="curated",
        title="t",
        patient_source_id="p1",
        patient_label="lbl",
        turns=(NormalizedTurn(role="main", prompt="Q?"),),
        message_history=({"role": "user", "content": "hi"},),
        original_answer="A.",
        original_evidence=({"tool_name": "x", "result_summary": "y"},),
        reviewer_guidance="guidance",
    )
    restored = NormalizedCase.from_payload(case.to_payload())
    assert restored == case
