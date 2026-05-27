from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from orm.models import Base, PatientSelectionEvent, AgentPatientAccess


def test_log_patient_selection_writes_event_and_access(monkeypatch):
    from orm import agent_logging_functions as alf

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    monkeypatch.setattr(alf, "SessionLocal", Session)

    alf.log_patient_selection(
        session_id="s1",
        user_id=1,
        source_id="src-1",
        display_name="Ann",
        selection_origin="user_click",
        action="selected",
        previous_source_id=None,
        run_id=None,
    )
    s = Session()
    assert s.query(PatientSelectionEvent).one().source_id == "src-1"
    assert s.query(AgentPatientAccess).filter_by(access_type="selection_chosen").one().source_id == "src-1"


def test_log_patient_clear_writes_clear_event_and_access(monkeypatch):
    from orm import agent_logging_functions as alf

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    monkeypatch.setattr(alf, "SessionLocal", Session)

    alf.log_patient_selection(
        session_id="s1",
        user_id=1,
        source_id=None,
        display_name=None,
        selection_origin="clear_button",
        action="cleared",
        previous_source_id="src-old",
        run_id=None,
    )
    s = Session()
    event = s.query(PatientSelectionEvent).one()
    assert event.action == "cleared"
    assert event.source_id is None
    assert event.previous_source_id == "src-old"
    access = s.query(AgentPatientAccess).filter_by(access_type="selection_cleared").one()
    assert access.source_id == "src-old"
