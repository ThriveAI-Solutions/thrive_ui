import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from orm.evaluation_models import (
    AdminNotification,
    AgentRunFeedback,
    AgentRunFeedbackEvent,
    EvaluationCase,
    EvaluationCaseResult,
    EvaluationReviewEvent,
    EvaluationRun,
)
from orm.models import AgentRun, Base, RoleTypeEnum, User, UserRole


def _session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _seed_user_and_run(session):
    role = UserRole(role_name="Doctor", description="Doctor", role=RoleTypeEnum.DOCTOR)
    session.add(role)
    session.flush()
    user = User(
        username="doctor",
        first_name="Test",
        last_name="Doctor",
        password="x",
        email="doctor@example.test",
        organization="Thrive",
        user_role_id=role.id,
    )
    session.add(user)
    session.flush()
    run = AgentRun(
        run_id="agent-run-1",
        session_id="session-1",
        group_id="group-1",
        user_id=user.id,
        user_role=1,
        status="success",
        success=True,
        logging_mode="full",
    )
    session.add(run)
    session.commit()
    return user, run


def test_feedback_and_audit_event_round_trip():
    session = _session()
    user, run = _seed_user_and_run(session)
    feedback = AgentRunFeedback(
        agent_run_id=run.id,
        user_id=user.id,
        rating="down",
        category="incorrect_answer",
        comment="The medication status is wrong.",
    )
    session.add(feedback)
    session.flush()
    session.add(
        AgentRunFeedbackEvent(
            feedback_id=feedback.id,
            actor_user_id=user.id,
            old_rating=None,
            new_rating="down",
            old_category=None,
            new_category="incorrect_answer",
            old_comment=None,
            new_comment="The medication status is wrong.",
        )
    )
    session.commit()

    saved_feedback = session.query(AgentRunFeedback).one()
    saved_event = session.query(AgentRunFeedbackEvent).one()
    assert saved_feedback.agent_run_id == run.id
    assert saved_feedback.user_id == user.id
    assert saved_event.new_rating == "down"


def test_feedback_owner_uniqueness_and_rating_check_constraints():
    session = _session()
    user, run = _seed_user_and_run(session)
    session.add(AgentRunFeedback(agent_run_id=run.id, user_id=user.id, rating="up"))
    session.commit()

    session.add(AgentRunFeedback(agent_run_id=run.id, user_id=user.id, rating="down"))
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()

    session.add(AgentRunFeedback(agent_run_id=run.id, user_id=user.id + 1, rating="sideways"))
    with pytest.raises(IntegrityError):
        session.commit()


def test_case_run_result_review_and_notification_round_trip():
    session = _session()
    user, _ = _seed_user_and_run(session)
    case = EvaluationCase(
        case_id="case-1",
        version=1,
        source_type="feedback",
        status="active",
        payload_json='{"question":"Which medications are active?"}',
        created_by=user.id,
    )
    run = EvaluationRun(
        run_id="eval-run-1",
        run_type="single_feedback",
        execution_mode="synchronous",
        status="queued",
        requested_by=user.id,
        total_cases=1,
    )
    session.add_all([case, run])
    session.flush()
    result = EvaluationCaseResult(
        evaluation_run_id=run.id,
        evaluation_case_id=case.id,
        ordinal=0,
        attempt=1,
        status="pending",
        result_json='{"answer":"rerun"}',
    )
    session.add(result)
    session.flush()
    session.add(
        EvaluationReviewEvent(
            result_id=result.id,
            reviewer_id=user.id,
            old_verdict=None,
            new_verdict="cant_tell",
            note="Needs clinical review.",
        )
    )
    session.add(
        AdminNotification(
            user_id=user.id,
            evaluation_run_id=run.id,
            kind="evaluation_completed",
        )
    )
    session.commit()

    assert session.query(EvaluationCase).one().payload_json == '{"question":"Which medications are active?"}'
    assert session.query(EvaluationRun).one().completed_cases == 0
    assert session.query(EvaluationCaseResult).one().attempt == 1
    assert session.query(EvaluationReviewEvent).one().new_verdict == "cant_tell"
    assert session.query(AdminNotification).one().read_at is None


def test_curated_promotion_draft_and_version_uniqueness_constraints():
    session = _session()
    user, _ = _seed_user_and_run(session)
    source_case = EvaluationCase(
        case_id="feedback-case-1",
        version=1,
        source_type="feedback",
        status="active",
        payload_json='{"schema_version":1,"question":"original"}',
        created_by=user.id,
    )
    session.add(source_case)
    session.flush()
    draft = EvaluationCase(
        case_id="curated-case-1",
        version=1,
        source_type="curated",
        status="draft",
        promoted_from_case_id=source_case.id,
        payload_json='{"schema_version":1,"question_template":"Which medications are active?"}',
        created_by=user.id,
    )
    session.add(draft)
    session.commit()

    saved_draft = session.query(EvaluationCase).filter_by(source_type="curated").one()
    assert saved_draft.status == "draft"
    assert saved_draft.promoted_from_case_id == source_case.id

    session.add(
        EvaluationCase(
            case_id="curated-case-1",
            version=1,
            source_type="curated",
            status="draft",
            payload_json='{"schema_version":1,"question_template":"duplicate"}',
            created_by=user.id,
        )
    )
    with pytest.raises(IntegrityError):
        session.commit()


def test_duplicate_case_result_attempt_is_rejected():
    session = _session()
    user, _ = _seed_user_and_run(session)
    case = EvaluationCase(
        case_id="case-1",
        version=1,
        source_type="feedback",
        status="active",
        payload_json='{"schema_version":1}',
        created_by=user.id,
    )
    run = EvaluationRun(
        run_id="eval-run-1",
        run_type="feedback_batch",
        execution_mode="asynchronous",
        status="running",
        requested_by=user.id,
        total_cases=1,
    )
    session.add_all([case, run])
    session.flush()
    session.add_all(
        [
            EvaluationCaseResult(
                evaluation_run_id=run.id,
                evaluation_case_id=case.id,
                ordinal=0,
                attempt=1,
                status="completed",
            ),
            EvaluationCaseResult(
                evaluation_run_id=run.id,
                evaluation_case_id=case.id,
                ordinal=0,
                attempt=1,
                status="pending",
            ),
        ]
    )

    with pytest.raises(IntegrityError):
        session.commit()


def test_agent_run_supports_minimal_disabled_mode_envelope():
    session = _session()
    user, _ = _seed_user_and_run(session)
    disabled_run = AgentRun(
        run_id="disabled-run-1",
        session_id="session-disabled",
        group_id="group-disabled",
        user_id=user.id,
        user_role=1,
        status="success",
        success=True,
        logging_mode="disabled",
    )
    session.add(disabled_run)
    session.commit()

    saved = session.query(AgentRun).filter_by(run_id="disabled-run-1").one()
    assert saved.question is None
    assert saved.selected_patient_source_id is None
    assert saved.final_answer_text is None


def test_evaluation_indexes_are_declared_for_catalog_run_result_and_notifications():
    session = _session()
    indexes_by_table = {
        table_name: {index["name"] for index in inspect(session.bind).get_indexes(table_name)}
        for table_name in (
            "thrive_agent_run_feedback",
            "thrive_evaluation_case",
            "thrive_evaluation_run",
            "thrive_evaluation_case_result",
            "thrive_admin_notification",
        )
    }

    assert "ix_thrive_agent_run_feedback_user" in indexes_by_table["thrive_agent_run_feedback"]
    assert "ix_thrive_evaluation_case_source_status" in indexes_by_table["thrive_evaluation_case"]
    assert "ix_thrive_evaluation_run_status_time" in indexes_by_table["thrive_evaluation_run"]
    assert "ix_thrive_evaluation_case_result_run_ordinal" in indexes_by_table["thrive_evaluation_case_result"]
    assert "ix_thrive_admin_notification_unread" in indexes_by_table["thrive_admin_notification"]
