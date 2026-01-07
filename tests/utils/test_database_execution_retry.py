"""Tests for Issue #30: Auto-retry SQL on database execution failure with full error context.

This test file verifies that:
1. Database execution errors trigger the auto-retry mechanism
2. The error message is captured and passed to generate_sql_retry()
3. The failed SQL is captured and passed to generate_sql_retry()
4. Retries stop after max_sql_retries attempts
5. Non-recoverable errors skip retries
"""

import types
from contextlib import nullcontext

import pandas as pd
import pytest


def _fake_st():
    """Create a fake streamlit module for testing."""
    st = types.SimpleNamespace()

    class _State(dict):
        def __getattr__(self, k):
            try:
                return self[k]
            except KeyError:
                raise AttributeError(k)

        def __setattr__(self, k, v):
            self[k] = v

    st.session_state = _State()
    st.session_state.update(
        {
            "messages": [],
            "show_sql": True,
            "show_table": True,
            "show_chart": False,
            "show_summary": False,
            "speak_summary": False,
        }
    )

    st.chat_message = lambda *_args, **_kwargs: nullcontext()

    class _Placeholder:
        def markdown(self, *args, **kwargs):
            pass

        def empty(self):
            pass

    st.empty = lambda: _Placeholder()

    class _Ctx:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    st.columns = lambda sizes: [_Ctx() for _ in sizes]
    st.button = lambda *a, **k: False
    st.write = lambda *a, **k: None
    st.markdown = lambda *a, **k: None
    st.code = lambda *a, **k: None
    st.error = lambda *a, **k: None
    st.warning = lambda *a, **k: None
    st.caption = lambda *a, **k: None
    st.info = lambda *a, **k: None
    st.text_input = lambda *a, **k: ""

    class StopException(Exception):
        pass

    st.StopException = StopException

    def _stop():
        raise StopException()

    st.stop = _stop
    st.expander = lambda *_a, **_k: nullcontext()
    st.dataframe = lambda *a, **k: None
    st.plotly_chart = lambda *a, **k: None
    st.popover = lambda *a, **k: nullcontext()
    st.text = lambda *a, **k: None
    st.rerun = lambda: None
    return st


class _MockVNWithDatabaseError:
    """Mock VannaService that simulates database execution errors."""

    def __init__(self, error_message: str, fail_count: int = 3, fake_st=None):
        """Initialize the mock.

        Args:
            error_message: The database error message to simulate
            fail_count: Number of times to fail before succeeding (default: always fail)
            fake_st: The fake streamlit module to use for session state
        """
        self._error_message = error_message
        self._fail_count = fail_count
        self._current_attempt = 0
        self._retry_calls = []  # Track calls to generate_sql_retry
        self._fake_st = fake_st

    def is_sql_valid(self, sql: str) -> bool:
        return True

    def generate_sql(self, question: str):
        return ("SELECT * FROM patients", 0.01)

    def generate_sql_retry(
        self,
        question: str,
        failed_sql=None,
        error_message=None,
        attempt_number=2,
        user_feedback=None,
    ):
        """Track retry calls and return a modified SQL."""
        self._retry_calls.append({
            "question": question,
            "failed_sql": failed_sql,
            "error_message": error_message,
            "attempt_number": attempt_number,
            "user_feedback": user_feedback,
        })
        return (f"SELECT * FROM patients /* retry attempt {attempt_number} */", 0.01)

    def run_sql(self, sql: str):
        """Simulate database execution - fails then optionally succeeds."""
        self._current_attempt += 1

        if self._current_attempt <= self._fail_count:
            # Fail - store error in the fake session state
            if self._fake_st is not None:
                self._fake_st.session_state["last_run_sql_error"] = self._error_message
                self._fake_st.session_state["last_failed_sql"] = sql
            return None  # Database error returns None
        else:
            # Succeed
            return pd.DataFrame({"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]})

    def should_generate_chart(self, question, sql, df):
        return False

    def generate_summary(self, question: str, df: pd.DataFrame):
        return ("Summary of results", 0.01)

    def generate_followup_questions(self, question: str, sql: str, df):
        return ["Follow-up 1", "Follow-up 2"]


class TestDatabaseExecutionRetry:
    """Test that database execution errors trigger auto-retry with full context."""

    def test_database_error_triggers_retry(self, monkeypatch):
        """Test that a database execution error triggers the auto-retry loop."""
        import utils.chat_bot_helper as cbh

        fake_st = _fake_st()
        monkeypatch.setattr(cbh, "st", fake_st)
        monkeypatch.setattr(cbh, "get_ethical_guideline", lambda q: ("", 1))
        monkeypatch.setattr(cbh.Message, "save", lambda self: self, raising=True)

        # Create mock that fails twice then succeeds on third attempt
        # Use a recoverable error (type mismatch is not in NON_RECOVERABLE_ERROR_PATTERNS)
        vn = _MockVNWithDatabaseError(
            error_message='type mismatch in comparison',
            fail_count=2,  # Fail first 2 attempts
            fake_st=fake_st,
        )
        monkeypatch.setattr(cbh, "get_vn", lambda: vn)

        cbh.set_question("Show all patients", render=False)
        cbh.normal_message_flow("Show all patients")

        # Verify retry was called with error context
        assert len(vn._retry_calls) >= 1, "generate_sql_retry should have been called"

        # First retry should have received error context
        first_retry = vn._retry_calls[0]
        assert first_retry["error_message"] == 'type mismatch in comparison'
        assert first_retry["failed_sql"] is not None
        assert "SELECT" in first_retry["failed_sql"]
        assert first_retry["attempt_number"] == 2

    def test_error_message_passed_to_retry(self, monkeypatch):
        """Test that the exact database error message is passed to generate_sql_retry."""
        import utils.chat_bot_helper as cbh

        fake_st = _fake_st()
        monkeypatch.setattr(cbh, "st", fake_st)
        monkeypatch.setattr(cbh, "get_ethical_guideline", lambda q: ("", 1))
        monkeypatch.setattr(cbh.Message, "save", lambda self: self, raising=True)

        # Use a recoverable error (division by zero is not in NON_RECOVERABLE_ERROR_PATTERNS)
        error_msg = 'division by zero'
        vn = _MockVNWithDatabaseError(error_message=error_msg, fail_count=2, fake_st=fake_st)
        monkeypatch.setattr(cbh, "get_vn", lambda: vn)

        cbh.set_question("Show data from table", render=False)
        cbh.normal_message_flow("Show data from table")

        # Verify the exact error message was passed
        assert len(vn._retry_calls) >= 1
        assert vn._retry_calls[0]["error_message"] == error_msg

    def test_failed_sql_passed_to_retry(self, monkeypatch):
        """Test that the failed SQL is passed to generate_sql_retry."""
        import utils.chat_bot_helper as cbh

        fake_st = _fake_st()
        monkeypatch.setattr(cbh, "st", fake_st)
        monkeypatch.setattr(cbh, "get_ethical_guideline", lambda q: ("", 1))
        monkeypatch.setattr(cbh.Message, "save", lambda self: self, raising=True)

        vn = _MockVNWithDatabaseError(
            error_message="syntax error",
            fail_count=2,
            fake_st=fake_st,
        )
        monkeypatch.setattr(cbh, "get_vn", lambda: vn)

        cbh.set_question("Get patient count", render=False)
        cbh.normal_message_flow("Get patient count")

        # Verify failed SQL was captured
        assert len(vn._retry_calls) >= 1
        assert vn._retry_calls[0]["failed_sql"] is not None
        assert "SELECT" in vn._retry_calls[0]["failed_sql"]

    def test_retry_stops_after_max_attempts(self, monkeypatch):
        """Test that retries stop after max_sql_retries attempts."""
        import utils.chat_bot_helper as cbh

        fake_st = _fake_st()
        monkeypatch.setattr(cbh, "st", fake_st)
        monkeypatch.setattr(cbh, "get_ethical_guideline", lambda q: ("", 1))
        monkeypatch.setattr(cbh.Message, "save", lambda self: self, raising=True)

        # Always fail - never succeed
        vn = _MockVNWithDatabaseError(
            error_message="persistent error",
            fail_count=100,  # Always fail
            fake_st=fake_st,
        )
        monkeypatch.setattr(cbh, "get_vn", lambda: vn)

        cbh.set_question("Impossible query", render=False)

        try:
            cbh.normal_message_flow("Impossible query")
        except fake_st.StopException:
            pass  # Expected - st.stop() called after retries exhausted

        # Default max_sql_retries is 2, so we should have 2 retry calls
        # (first attempt uses generate_sql, retries 2 and 3 use generate_sql_retry)
        assert len(vn._retry_calls) == 2, f"Expected 2 retry calls, got {len(vn._retry_calls)}"

    def test_progressive_attempt_numbers(self, monkeypatch):
        """Test that attempt numbers increase with each retry."""
        import utils.chat_bot_helper as cbh

        fake_st = _fake_st()
        monkeypatch.setattr(cbh, "st", fake_st)
        monkeypatch.setattr(cbh, "get_ethical_guideline", lambda q: ("", 1))
        monkeypatch.setattr(cbh.Message, "save", lambda self: self, raising=True)

        vn = _MockVNWithDatabaseError(
            error_message="keep failing",
            fail_count=100,
            fake_st=fake_st,
        )
        monkeypatch.setattr(cbh, "get_vn", lambda: vn)

        cbh.set_question("Query with retries", render=False)

        try:
            cbh.normal_message_flow("Query with retries")
        except fake_st.StopException:
            pass

        # Check attempt numbers are progressive (2, 3, ...)
        assert len(vn._retry_calls) >= 2
        assert vn._retry_calls[0]["attempt_number"] == 2
        assert vn._retry_calls[1]["attempt_number"] == 3


class TestNonRecoverableErrors:
    """Test that non-recoverable errors skip the retry loop."""

    @pytest.mark.parametrize("error_pattern", [
        'relation "missing_table" does not exist',
        'table "missing" does not exist',
        'column "bad_col" does not exist',
        "permission denied for table users",
        "access denied",
        "authentication failed",
        'database "nope" does not exist',
        'schema "missing" does not exist',
    ])
    def test_non_recoverable_error_skips_retry(self, error_pattern, monkeypatch):
        """Test that non-recoverable errors skip the auto-retry loop."""
        import utils.chat_bot_helper as cbh

        fake_st = _fake_st()
        monkeypatch.setattr(cbh, "st", fake_st)
        monkeypatch.setattr(cbh, "get_ethical_guideline", lambda q: ("", 1))
        monkeypatch.setattr(cbh.Message, "save", lambda self: self, raising=True)

        vn = _MockVNWithDatabaseError(
            error_message=error_pattern,
            fail_count=100,
            fake_st=fake_st,
        )
        monkeypatch.setattr(cbh, "get_vn", lambda: vn)

        cbh.set_question("Query that hits non-recoverable error", render=False)

        try:
            cbh.normal_message_flow("Query that hits non-recoverable error")
        except fake_st.StopException:
            pass

        # For non-recoverable errors, retry should NOT be called
        # The first run_sql fails and is detected as non-recoverable
        assert len(vn._retry_calls) == 0, (
            f"Non-recoverable error '{error_pattern}' should skip retries, "
            f"but got {len(vn._retry_calls)} retry calls"
        )


class TestRetryUIFeedback:
    """Test that retry attempts show appropriate UI feedback."""

    def test_retry_shows_attempt_number(self, monkeypatch):
        """Test that retry attempts display the attempt number to the user."""
        import utils.chat_bot_helper as cbh

        fake_st = _fake_st()
        info_calls = []

        def capture_info(msg):
            info_calls.append(msg)

        fake_st.info = capture_info
        monkeypatch.setattr(cbh, "st", fake_st)
        monkeypatch.setattr(cbh, "get_ethical_guideline", lambda q: ("", 1))
        monkeypatch.setattr(cbh.Message, "save", lambda self: self, raising=True)

        vn = _MockVNWithDatabaseError(
            error_message="temporary error",
            fail_count=2,  # Fail twice, succeed on third
            fake_st=fake_st,
        )
        monkeypatch.setattr(cbh, "get_vn", lambda: vn)

        cbh.set_question("Query with retry feedback", render=False)
        cbh.normal_message_flow("Query with retry feedback")

        # Check that retry status was shown
        retry_info = [msg for msg in info_calls if "Attempt" in msg]
        assert len(retry_info) >= 1, "Should show retry attempt info to user"
        assert "2/" in retry_info[0] or "Attempt 2" in retry_info[0]


class TestErrorDetailsInExpandable:
    """Test that error details are shown in expandable sections."""

    def test_error_shown_after_retries_exhausted(self, monkeypatch):
        """Test that error details are available after all retries fail."""
        import utils.chat_bot_helper as cbh

        fake_st = _fake_st()
        monkeypatch.setattr(cbh, "st", fake_st)
        monkeypatch.setattr(cbh, "get_ethical_guideline", lambda q: ("", 1))
        monkeypatch.setattr(cbh.Message, "save", lambda self: self, raising=True)

        error_msg = "type mismatch: cannot compare text with integer"
        vn = _MockVNWithDatabaseError(
            error_message=error_msg,
            fail_count=100,
            fake_st=fake_st,
        )
        monkeypatch.setattr(cbh, "get_vn", lambda: vn)

        cbh.set_question("Query that always fails", render=False)

        try:
            cbh.normal_message_flow("Query that always fails")
        except fake_st.StopException:
            pass

        # Verify error context is stored in session state for UI display
        assert fake_st.session_state.get("pending_sql_error") is True
        assert fake_st.session_state.get("retry_error_msg") == error_msg
        assert fake_st.session_state.get("retry_failed_sql") is not None


class TestVannaServiceRunSql:
    """Test that VannaService.run_sql properly captures error context."""

    def test_run_sql_stores_error_in_session_state(self, monkeypatch):
        """Test that run_sql stores the error message in session state on failure."""
        from unittest.mock import MagicMock, patch
        from utils.vanna_calls import VannaService

        # Create a VannaService with mocked vn
        with patch.object(VannaService, "_setup_vanna"):
            from utils.vanna_calls import UserContext
            service = VannaService.__new__(VannaService)
            service.vn = MagicMock()
            service.vn.run_sql.side_effect = Exception("Connection refused")

        # Mock session state
        class MockSessionState(dict):
            def __getattr__(self, k):
                try:
                    return self[k]
                except KeyError:
                    raise AttributeError(k)

            def __setattr__(self, k, v):
                self[k] = v

        mock_session = MockSessionState()

        with patch("streamlit.session_state", mock_session):
            with patch("streamlit.error"):
                with patch("streamlit.spinner", return_value=nullcontext()):
                    # Call run_sql directly (bypass cache)
                    result = VannaService.run_sql.__wrapped__(service, "SELECT 1")

        # Verify error was stored
        assert result is None
        assert mock_session.get("last_run_sql_error") == "Connection refused"
        assert mock_session.get("last_failed_sql") == "SELECT 1"

    def test_run_sql_clears_error_on_success(self, monkeypatch):
        """Test that run_sql clears previous error context on success."""
        from unittest.mock import MagicMock, patch
        from utils.vanna_calls import VannaService

        with patch.object(VannaService, "_setup_vanna"):
            from utils.vanna_calls import UserContext
            service = VannaService.__new__(VannaService)
            service.vn = MagicMock()
            service.vn.run_sql.return_value = pd.DataFrame({"col": [1]})

        class MockSessionState(dict):
            def __getattr__(self, k):
                try:
                    return self[k]
                except KeyError:
                    raise AttributeError(k)

            def __setattr__(self, k, v):
                self[k] = v

        mock_session = MockSessionState()
        mock_session["last_run_sql_error"] = "previous error"
        mock_session["last_failed_sql"] = "previous sql"

        with patch("streamlit.session_state", mock_session):
            with patch("streamlit.spinner", return_value=nullcontext()):
                result = VannaService.run_sql.__wrapped__(service, "SELECT 1")

        # Verify previous error was cleared
        assert result is not None
        assert mock_session.get("last_run_sql_error") is None
        assert mock_session.get("last_failed_sql") is None
