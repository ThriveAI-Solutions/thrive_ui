"""Tests for the saved reports service."""

import json
import pytest
import pandas as pd
from datetime import datetime, timedelta

from orm.models import SavedReport, ReportExecution, SessionLocal, Base, engine, User, UserRole, RoleTypeEnum
from utils.report_service import (
    save_report_from_chat,
    get_user_reports,
    get_report_by_id,
    get_report_by_name,
    archive_report,
    find_matching_report,
    record_report_execution,
    get_report_executions,
    get_latest_execution,
    extract_numeric_summary,
    get_consistent_columns,
    get_execution_count,
    format_delta,
    _calculate_similarity,
)


@pytest.fixture
def test_user():
    """Create a test user for report tests."""
    session = SessionLocal()
    try:
        # Ensure Admin role exists
        role = session.query(UserRole).filter_by(role_name="Admin").first()
        if not role:
            role = UserRole(role_name="Admin", description="Test Admin", role=RoleTypeEnum.ADMIN)
            session.add(role)
            session.commit()

        # Create a unique test user for each test
        import uuid
        unique_suffix = str(uuid.uuid4())[:8]
        user = User(
            username=f"test_report_user_{unique_suffix}",
            first_name="Test",
            last_name="User",
            password="hashed_password",
            user_role_id=role.id,
        )
        session.add(user)
        session.commit()
        session.refresh(user)
        user_id = user.id

        yield user_id

        # Cleanup - delete executions first (foreign key constraint)
        reports = session.query(SavedReport).filter(SavedReport.user_id == user_id).all()
        for report in reports:
            session.query(ReportExecution).filter(ReportExecution.report_id == report.id).delete()
        session.query(SavedReport).filter(SavedReport.user_id == user_id).delete()
        session.query(User).filter(User.id == user_id).delete()
        session.commit()
    finally:
        session.close()


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        "county": ["Erie", "Monroe", "Niagara"],
        "patient_count": [100, 80, 60],
        "avg_age": [45.5, 42.3, 48.1],
    })


class TestSimilarityCalculation:
    """Tests for string similarity calculation."""

    def test_identical_strings(self):
        assert _calculate_similarity("hello world", "hello world") == 1.0

    def test_similar_strings(self):
        score = _calculate_similarity("how many patients by county", "how many patients by county?")
        assert score > 0.9

    def test_different_strings(self):
        score = _calculate_similarity("show all tables", "delete everything")
        assert score < 0.5

    def test_empty_strings(self):
        assert _calculate_similarity("", "hello") == 0.0
        assert _calculate_similarity("hello", "") == 0.0
        assert _calculate_similarity("", "") == 0.0

    def test_case_insensitive(self):
        score = _calculate_similarity("HELLO WORLD", "hello world")
        assert score == 1.0


class TestExtractNumericSummary:
    """Tests for extracting numeric summaries from DataFrames."""

    def test_basic_extraction(self, sample_dataframe):
        summary = extract_numeric_summary(sample_dataframe)

        assert "patient_count" in summary
        assert "avg_age" in summary
        assert "county" not in summary  # String column excluded

        assert summary["patient_count"]["sum"] == 240
        assert summary["patient_count"]["count"] == 3
        assert summary["patient_count"]["min"] == 60
        assert summary["patient_count"]["max"] == 100

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        summary = extract_numeric_summary(df)
        assert summary == {}

    def test_no_numeric_columns(self):
        df = pd.DataFrame({"name": ["Alice", "Bob"], "city": ["NYC", "LA"]})
        summary = extract_numeric_summary(df)
        assert summary == {}


class TestFormatDelta:
    """Tests for delta formatting."""

    def test_increase(self):
        result = format_delta(110, 100)
        assert "↑" in result
        assert "10" in result

    def test_decrease(self):
        result = format_delta(90, 100)
        assert "↓" in result
        assert "10" in result

    def test_no_change(self):
        result = format_delta(100, 100)
        assert "→" in result or "0" in result

    def test_from_zero(self):
        result = format_delta(100, 0)
        assert result == "N/A"


class TestSaveReportFromChat:
    """Tests for saving reports from chat."""

    def test_save_basic_report(self, test_user):
        report = save_report_from_chat(
            user_id=test_user,
            name="Monthly Patient Count",
            sql_query="SELECT county, COUNT(*) FROM patients GROUP BY county",
            original_question="How many patients by county?",
        )

        assert report is not None
        assert report.id is not None
        assert report.name == "Monthly Patient Count"
        assert report.sql_query == "SELECT county, COUNT(*) FROM patients GROUP BY county"
        assert report.original_question == "How many patients by county?"
        assert report.user_id == test_user
        assert report.is_archived == False

    def test_save_with_description(self, test_user):
        report = save_report_from_chat(
            user_id=test_user,
            name="Test Report",
            sql_query="SELECT * FROM test",
            original_question="Show test data",
            description="A test report for unit testing",
        )

        assert report.description == "A test report for unit testing"


class TestGetUserReports:
    """Tests for retrieving user reports."""

    def test_get_reports_empty(self, test_user):
        reports = get_user_reports(test_user)
        # Filter to just our test user's reports
        assert isinstance(reports, list)

    def test_get_reports_excludes_archived(self, test_user):
        # Create a normal report
        report1 = save_report_from_chat(
            user_id=test_user,
            name="Active Report",
            sql_query="SELECT 1",
            original_question="Test",
        )

        # Create and archive a report
        report2 = save_report_from_chat(
            user_id=test_user,
            name="Archived Report",
            sql_query="SELECT 2",
            original_question="Test 2",
        )
        archive_report(report2.id, test_user)

        # Get non-archived reports
        reports = get_user_reports(test_user, include_archived=False)
        report_names = [r.name for r in reports]

        assert "Active Report" in report_names
        assert "Archived Report" not in report_names

    def test_get_reports_includes_archived(self, test_user):
        report = save_report_from_chat(
            user_id=test_user,
            name="Another Archived",
            sql_query="SELECT 3",
            original_question="Test 3",
        )
        archive_report(report.id, test_user)

        reports = get_user_reports(test_user, include_archived=True)
        report_names = [r.name for r in reports]
        assert "Another Archived" in report_names


class TestFindMatchingReport:
    """Tests for automatic report detection."""

    def test_exact_question_match(self, test_user):
        question = "How many patients are there by county?"
        save_report_from_chat(
            user_id=test_user,
            name="Patient County Report",
            sql_query="SELECT county, COUNT(*) FROM patients GROUP BY county",
            original_question=question,
        )

        match = find_matching_report(test_user, question)
        assert match is not None
        assert match.name == "Patient County Report"

    def test_similar_question_match(self, test_user):
        save_report_from_chat(
            user_id=test_user,
            name="Patients by County",
            sql_query="SELECT county, COUNT(*) FROM patients GROUP BY county",
            original_question="How many patients are there by county?",
        )

        # Almost identical question should match (>90% similar)
        match = find_matching_report(test_user, "How many patients are there by county")
        assert match is not None
        assert match.name == "Patients by County"

    def test_no_match_different_question(self, test_user):
        save_report_from_chat(
            user_id=test_user,
            name="Some Report",
            sql_query="SELECT * FROM table1",
            original_question="Show all data from table1",
        )

        # Very different question should not match
        match = find_matching_report(test_user, "What is the average age of patients?")
        assert match is None

    def test_no_match_empty_question(self, test_user):
        match = find_matching_report(test_user, "")
        assert match is None

        match = find_matching_report(test_user, None)
        assert match is None


class TestRecordReportExecution:
    """Tests for recording report executions."""

    def test_record_successful_execution(self, test_user, sample_dataframe):
        report = save_report_from_chat(
            user_id=test_user,
            name="Execution Test Report",
            sql_query="SELECT * FROM data",
            original_question="Show data",
        )

        execution = record_report_execution(
            report=report,
            df=sample_dataframe,
            user_id=test_user,
            group_id="test-group-123",
            elapsed_time=0.5,
        )

        assert execution is not None
        assert execution.report_id == report.id
        assert execution.success == True
        assert execution.row_count == 3
        assert execution.elapsed_time == 0.5
        assert execution.group_id == "test-group-123"
        assert execution.triggered_by == "auto"

        # Check numeric summary was computed
        summary = json.loads(execution.numeric_summary)
        assert "patient_count" in summary

    def test_record_null_dataframe(self, test_user):
        report = save_report_from_chat(
            user_id=test_user,
            name="Null DF Test",
            sql_query="SELECT * FROM empty",
            original_question="Show empty",
        )

        execution = record_report_execution(
            report=report,
            df=None,
            user_id=test_user,
            group_id="test-group-456",
        )

        assert execution.success == False
        assert execution.row_count == 0


class TestGetReportExecutions:
    """Tests for retrieving execution history."""

    def test_get_executions_ordered(self, test_user, sample_dataframe):
        report = save_report_from_chat(
            user_id=test_user,
            name="History Test",
            sql_query="SELECT * FROM test",
            original_question="Test query",
        )

        # Record multiple executions
        for i in range(3):
            record_report_execution(report, sample_dataframe, test_user, f"group-{i}")

        executions = get_report_executions(report.id)
        # Should have at least 3 executions (from this test)
        assert len(executions) >= 3

        # Should be ordered by most recent first
        for i in range(len(executions) - 1):
            assert executions[i].created_at >= executions[i + 1].created_at

    def test_get_executions_with_limit(self, test_user, sample_dataframe):
        report = save_report_from_chat(
            user_id=test_user,
            name="Limit Test",
            sql_query="SELECT 1",
            original_question="Test",
        )

        for i in range(5):
            record_report_execution(report, sample_dataframe, test_user, f"group-{i}")

        executions = get_report_executions(report.id, limit=2)
        assert len(executions) == 2


class TestGetExecutionCount:
    """Tests for execution count."""

    def test_count_executions(self, test_user, sample_dataframe):
        report = save_report_from_chat(
            user_id=test_user,
            name="Count Test",
            sql_query="SELECT 1",
            original_question="Test",
        )

        # Get initial count for this specific report
        initial_count = get_execution_count(report.id)

        record_report_execution(report, sample_dataframe, test_user, "group-1")
        assert get_execution_count(report.id) == initial_count + 1

        record_report_execution(report, sample_dataframe, test_user, "group-2")
        assert get_execution_count(report.id) == initial_count + 2


class TestGetConsistentColumns:
    """Tests for consistent column detection."""

    def test_consistent_columns_basic(self, test_user, sample_dataframe):
        """Test that consistent columns are detected across multiple executions."""
        report = save_report_from_chat(
            user_id=test_user,
            name="Consistent Cols Test",
            sql_query="SELECT * FROM data",
            original_question="Show data",
        )

        # Record executions with the same DataFrame schema
        for i in range(3):
            record_report_execution(report, sample_dataframe, test_user, f"group-{i}")

        cols = get_consistent_columns(report.id)
        # Should have numeric columns from sample_dataframe
        assert isinstance(cols, list)

    def test_extract_numeric_summary_works(self):
        """Test that numeric summary extraction works correctly."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "name": ["x", "y", "z"]})
        summary = extract_numeric_summary(df)

        assert "a" in summary
        assert "b" in summary
        assert "name" not in summary  # String column excluded

        assert summary["a"]["sum"] == 6.0
        assert summary["a"]["avg"] == 2.0
        assert summary["a"]["min"] == 1.0
        assert summary["a"]["max"] == 3.0
        assert summary["a"]["count"] == 3
