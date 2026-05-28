from orm.models import ToolCall


def test_tool_call_columns_present():
    tc = ToolCall(
        session_id="abc",
        user_id=1,
        user_role=1,
        selected_patient_source_id="src-123",
        tool_name="find_patient",
        arguments_json='{"first_name":"John"}',
        result_summary="row_count=5",
        elapsed_ms=42,
        success=True,
        error=None,
    )
    assert tc.tool_name == "find_patient"
    assert tc.success is True
    assert tc.selected_patient_source_id == "src-123"


def test_tool_call_allows_null_selected_patient():
    tc = ToolCall(
        session_id="abc",
        user_id=1,
        user_role=1,
        selected_patient_source_id=None,
        tool_name="search_patients_by_criteria",
        arguments_json="{}",
        result_summary="row_count=10",
        elapsed_ms=100,
        success=True,
    )
    assert tc.selected_patient_source_id is None
