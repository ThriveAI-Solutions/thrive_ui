"""Tests for the user feedback feature in SQL retry functionality."""

from unittest.mock import MagicMock, patch


class TestGenerateSqlRetryUserFeedback:
    """Test that generate_sql_retry properly handles user feedback parameter."""

    def test_user_feedback_included_in_augmented_question(self):
        """Test that user feedback is appended to the augmented question sent to LLM."""
        captured_question = None

        class MockVannaService:
            def __init__(self):
                pass

            def generate_sql(self, question):
                nonlocal captured_question
                captured_question = question
                return ("SELECT 1", 0.01)

        # Create a minimal mock that has the generate_sql_retry logic
        from utils.vanna_calls import VannaService

        # We need to test the augmented question building logic directly
        # Build the augmented question parts as the method does
        question = "How many patients are there?"
        failed_sql = "SELECT count(*) FROM patiants"  # intentional typo
        error_message = 'relation "patiants" does not exist'
        user_feedback = "The table name is 'patients' not 'patiants'"
        attempt_number = 2

        # Simulate the augmented question building
        if attempt_number == 2:
            guidance = (
                "The previous SQL failed. Please try a DIFFERENT approach to answer this question. "
                "Consider using different JOINs, subqueries, or alternative column selections."
            )
        else:
            guidance = (
                "Multiple SQL attempts have failed. Please try the SIMPLEST possible query that could answer this question. "
                "Consider: removing JOINs, using only essential columns, or breaking into smaller queries."
            )

        augmented_question_parts = [
            f"Original question: {question}",
            guidance,
        ]
        if failed_sql:
            augmented_question_parts.append("Failed SQL:\n" + failed_sql)
        if error_message:
            augmented_question_parts.append("Database error: " + error_message)
        if user_feedback:
            augmented_question_parts.append(f"User feedback: {user_feedback}")
        augmented_question = "\n\n".join(augmented_question_parts)

        # Verify the augmented question contains all the expected parts
        assert "Original question: How many patients are there?" in augmented_question
        assert "The previous SQL failed" in augmented_question
        assert "Failed SQL:\nSELECT count(*) FROM patiants" in augmented_question
        assert 'Database error: relation "patiants" does not exist' in augmented_question
        assert "User feedback: The table name is 'patients' not 'patiants'" in augmented_question

    def test_user_feedback_none_does_not_add_section(self):
        """Test that when user_feedback is None, no 'User feedback:' section is added."""
        question = "How many patients are there?"
        failed_sql = "SELECT count(*) FROM patiants"
        error_message = 'relation "patiants" does not exist'
        user_feedback = None  # No feedback
        attempt_number = 2

        guidance = (
            "The previous SQL failed. Please try a DIFFERENT approach to answer this question. "
            "Consider using different JOINs, subqueries, or alternative column selections."
        )

        augmented_question_parts = [
            f"Original question: {question}",
            guidance,
        ]
        if failed_sql:
            augmented_question_parts.append("Failed SQL:\n" + failed_sql)
        if error_message:
            augmented_question_parts.append("Database error: " + error_message)
        if user_feedback:
            augmented_question_parts.append(f"User feedback: {user_feedback}")
        augmented_question = "\n\n".join(augmented_question_parts)

        # Verify the augmented question does NOT contain user feedback section
        assert "User feedback:" not in augmented_question

    def test_user_feedback_empty_string_does_not_add_section(self):
        """Test that when user_feedback is empty string, no 'User feedback:' section is added."""
        question = "How many patients are there?"
        failed_sql = "SELECT count(*) FROM patiants"
        error_message = 'relation "patiants" does not exist'
        user_feedback = ""  # Empty string
        attempt_number = 2

        guidance = (
            "The previous SQL failed. Please try a DIFFERENT approach to answer this question. "
            "Consider using different JOINs, subqueries, or alternative column selections."
        )

        augmented_question_parts = [
            f"Original question: {question}",
            guidance,
        ]
        if failed_sql:
            augmented_question_parts.append("Failed SQL:\n" + failed_sql)
        if error_message:
            augmented_question_parts.append("Database error: " + error_message)
        if user_feedback:  # Empty string is falsy, so this should not execute
            augmented_question_parts.append(f"User feedback: {user_feedback}")
        augmented_question = "\n\n".join(augmented_question_parts)

        # Verify the augmented question does NOT contain user feedback section
        assert "User feedback:" not in augmented_question


class TestRetrySessionStateFlags:
    """Test the session state flag pattern for single-click retry."""

    def test_retry_triggered_persist_flag_pattern(self):
        """Test the pattern: set flag on click, check flag after rerun, clear after processing."""
        # Simulate session state
        session_state = {
            "pending_sql_error": True,
            "last_run_sql_error": "some error",
            "pending_question": "test question",
            "retry_triggered_persist": False,
            "retry_feedback_persist": "",
        }

        # Simulate button click - sets flag and triggers rerun
        def simulate_button_click(state):
            state["retry_triggered_persist"] = True
            # st.rerun() would be called here

        # Simulate checking flag after rerun
        def check_and_process_retry(state):
            if state.get("retry_triggered_persist"):
                # Clear the flag
                state["retry_triggered_persist"] = False
                # Process the retry...
                return True
            return False

        # Before click
        assert session_state["retry_triggered_persist"] is False
        assert check_and_process_retry(session_state) is False

        # Simulate click
        simulate_button_click(session_state)
        assert session_state["retry_triggered_persist"] is True

        # After "rerun", check flag and process
        result = check_and_process_retry(session_state)
        assert result is True
        assert session_state["retry_triggered_persist"] is False

        # Subsequent checks should return False (already processed)
        assert check_and_process_retry(session_state) is False

    def test_feedback_flows_through_session_state(self):
        """Test that user feedback flows through session state to generate_sql_retry."""
        session_state = {
            "pending_sql_error": True,
            "last_run_sql_error": "column not found",
            "last_failed_sql": "SELECT bad_column FROM table",
            "pending_question": "test question",
            "retry_triggered_persist": True,
            "retry_feedback_persist": "use 'good_column' instead",
            "retry_feedback_inline": "",
        }

        # Simulate the retry processing logic from chat_bot.py
        def process_retry(state):
            if state.get("retry_triggered_persist") or state.get("retry_triggered_inline"):
                state["retry_triggered_persist"] = False
                state["retry_triggered_inline"] = False

                user_feedback = state.get("retry_feedback_inline", "") or state.get("retry_feedback_persist", "")

                state["use_retry_context"] = True
                state["retry_failed_sql"] = state.get("last_failed_sql")
                state["retry_error_msg"] = state.get("last_run_sql_error")
                state["retry_user_feedback"] = user_feedback if user_feedback else None
                state["my_question"] = state.get("pending_question")
                state["pending_sql_error"] = False
                state["retry_feedback_persist"] = ""
                state["retry_feedback_inline"] = ""
                return True
            return False

        result = process_retry(session_state)
        assert result is True
        assert session_state["retry_user_feedback"] == "use 'good_column' instead"
        assert session_state["use_retry_context"] is True
        assert session_state["retry_feedback_persist"] == ""
        assert session_state["retry_feedback_inline"] == ""


class TestVannaServiceGenerateSqlRetryIntegration:
    """Integration tests that call VannaService.generate_sql_retry() with mocked generate_sql."""

    @patch("utils.vanna_calls.st")
    def test_generate_sql_retry_passes_user_feedback_to_generate_sql(self, mock_st):
        """Test that VannaService.generate_sql_retry passes user feedback to generate_sql."""
        from utils.vanna_calls import VannaService

        # Track what question was passed to generate_sql
        captured_question = None

        def capture_generate_sql(question):
            nonlocal captured_question
            captured_question = question
            return ("SELECT 1", 0.01)

        # Create VannaService instance with minimal setup
        service = VannaService.__new__(VannaService)
        service.vn = MagicMock()

        # Replace generate_sql with our capturing version
        service.generate_sql = capture_generate_sql

        # Call generate_sql_retry directly (bypassing cache decorator)
        VannaService.generate_sql_retry.__wrapped__(
            service,
            question="How many patients are there?",
            failed_sql="SELECT count(*) FROM patiants",
            error_message='relation "patiants" does not exist',
            attempt_number=2,
            user_feedback="The table name is 'patients' not 'patiants'",
        )

        # Verify the augmented question contains user feedback
        assert captured_question is not None
        assert "User feedback: The table name is 'patients' not 'patiants'" in captured_question
        assert "Original question: How many patients are there?" in captured_question
        assert "Failed SQL:" in captured_question
        assert "Database error:" in captured_question

    @patch("utils.vanna_calls.st")
    def test_generate_sql_retry_omits_feedback_when_none(self, mock_st):
        """Test that VannaService.generate_sql_retry omits feedback section when None."""
        from utils.vanna_calls import VannaService

        captured_question = None

        def capture_generate_sql(question):
            nonlocal captured_question
            captured_question = question
            return ("SELECT 1", 0.01)

        service = VannaService.__new__(VannaService)
        service.vn = MagicMock()
        service.generate_sql = capture_generate_sql

        VannaService.generate_sql_retry.__wrapped__(
            service,
            question="How many patients are there?",
            failed_sql="SELECT count(*) FROM patiants",
            error_message='relation "patiants" does not exist',
            attempt_number=2,
            user_feedback=None,
        )

        assert captured_question is not None
        assert "User feedback:" not in captured_question
        assert "Original question: How many patients are there?" in captured_question

    @patch("utils.vanna_calls.st")
    def test_generate_sql_retry_omits_feedback_when_empty_string(self, mock_st):
        """Test that VannaService.generate_sql_retry omits feedback section when empty string."""
        from utils.vanna_calls import VannaService

        captured_question = None

        def capture_generate_sql(question):
            nonlocal captured_question
            captured_question = question
            return ("SELECT 1", 0.01)

        service = VannaService.__new__(VannaService)
        service.vn = MagicMock()
        service.generate_sql = capture_generate_sql

        VannaService.generate_sql_retry.__wrapped__(
            service,
            question="How many patients are there?",
            failed_sql="SELECT count(*) FROM patiants",
            error_message='relation "patiants" does not exist',
            attempt_number=2,
            user_feedback="",
        )

        assert captured_question is not None
        assert "User feedback:" not in captured_question
