import pytest
from datetime import date, datetime
from orm.models import RoleTypeEnum
from agent.deps import AgentDeps, SelectedPatient


def test_selected_patient_required_fields():
    sp = SelectedPatient(
        source_id="src-123",
        display_name="John Smith",
        dob=date(1962, 5, 1),
        selected_at=datetime.now(),
        selection_origin="user_click",
    )
    assert sp.source_id == "src-123"
    assert sp.selection_origin == "user_click"


def test_selected_patient_origin_validates():
    with pytest.raises((ValueError, TypeError)):
        SelectedPatient(
            source_id="src-123",
            display_name="John Smith",
            dob=None,
            selected_at=datetime.now(),
            selection_origin="hacked",
        )


def test_agent_deps_construction():
    deps = AgentDeps(
        user_id=1,
        user_role=RoleTypeEnum.DOCTOR,
        session_id="session-abc",
        selected_patient=None,
        last_dataframe=None,
        last_sql=None,
        last_query_meta=None,
        analytics_db=None,
        rag=None,
        sqlite_session=None,
        audit_logger=None,
    )
    assert deps.user_id == 1
    assert deps.selected_patient is None
