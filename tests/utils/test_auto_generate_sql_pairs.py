"""Tests for auto_generate_sql_pairs functionality in utils/vanna_calls.py."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Prompt parsing tests
# ---------------------------------------------------------------------------


class TestParseQuestionSqlResponse:
    """Tests for _parse_question_sql_response helper."""

    def test_valid_format(self):
        from utils.vanna_calls import _parse_question_sql_response

        text = (
            "QUESTION: How many patients are in the database?\n"
            "SQL: SELECT COUNT(*) FROM patients;"
        )
        question, sql = _parse_question_sql_response(text)
        assert question == "How many patients are in the database?"
        assert sql == "SELECT COUNT(*) FROM patients;"

    def test_valid_multiline_sql(self):
        from utils.vanna_calls import _parse_question_sql_response

        text = (
            "QUESTION: What are the top diagnoses?\n"
            "SQL: SELECT diagnosis, COUNT(*) AS cnt\n"
            "FROM encounters\n"
            "GROUP BY diagnosis\n"
            "ORDER BY cnt DESC\n"
            "LIMIT 10;"
        )
        question, sql = _parse_question_sql_response(text)
        assert question is not None
        assert "SELECT" in sql
        assert "GROUP BY" in sql

    def test_sql_with_markdown_fencing(self):
        from utils.vanna_calls import _parse_question_sql_response

        text = (
            "QUESTION: Count rows\n"
            "SQL: ```sql\nSELECT COUNT(*) FROM t;\n```"
        )
        question, sql = _parse_question_sql_response(text)
        assert question == "Count rows"
        assert sql == "SELECT COUNT(*) FROM t;"

    def test_missing_question_returns_none(self):
        from utils.vanna_calls import _parse_question_sql_response

        text = "SQL: SELECT 1;"
        question, sql = _parse_question_sql_response(text)
        assert question is None
        assert sql is None

    def test_missing_sql_returns_none(self):
        from utils.vanna_calls import _parse_question_sql_response

        text = "QUESTION: What is life?"
        question, sql = _parse_question_sql_response(text)
        assert question is None
        assert sql is None

    def test_empty_string_returns_none(self):
        from utils.vanna_calls import _parse_question_sql_response

        question, sql = _parse_question_sql_response("")
        assert question is None
        assert sql is None

    def test_case_insensitive_labels(self):
        from utils.vanna_calls import _parse_question_sql_response

        text = "question: What?\nsql: SELECT 1;"
        question, sql = _parse_question_sql_response(text)
        assert question == "What?"
        assert sql == "SELECT 1;"


# ---------------------------------------------------------------------------
# Prompt constants tests
# ---------------------------------------------------------------------------


class TestPromptConstants:
    """Verify prompt constants exist and contain expected keywords."""

    def test_generation_prompt_exists(self):
        from utils.vanna_calls import _SQL_PAIR_GENERATION_SYSTEM_PROMPT

        assert "QUESTION:" in _SQL_PAIR_GENERATION_SYSTEM_PROMPT
        assert "SQL:" in _SQL_PAIR_GENERATION_SYSTEM_PROMPT
        assert "SELECT" in _SQL_PAIR_GENERATION_SYSTEM_PROMPT

    def test_review_prompt_exists(self):
        from utils.vanna_calls import _SQL_PAIR_REVIEW_SYSTEM_PROMPT

        assert "PASS" in _SQL_PAIR_REVIEW_SYSTEM_PROMPT
        assert "FAIL" in _SQL_PAIR_REVIEW_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Integration tests with mocked LLM
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_vanna_service():
    """Create a mocked VannaService for auto_generate_sql_pairs tests."""
    mock_service = MagicMock()
    mock_service.vn = MagicMock()
    mock_service.get_training_data.return_value = pd.DataFrame(
        {"training_data_type": ["sql"], "question": ["Existing question"], "content": ["SELECT 1"]}
    )
    mock_service.vn.get_related_ddl.return_value = ["CREATE TABLE patients (id INT, name TEXT);"]
    return mock_service


@pytest.fixture
def mock_security_validator():
    """Mock security validator that passes everything."""
    mock_sv = MagicMock()
    mock_sv.validate_sql_content.return_value = (True, [])
    return mock_sv


class TestAutoGenerateSqlPairs:
    """Integration tests for auto_generate_sql_pairs with mocked dependencies."""

    @patch("utils.vanna_calls.st")
    @patch("utils.vanna_calls.write_to_file_and_training")
    @patch("utils.vanna_calls.VannaService.from_streamlit_session")
    def test_successful_generation(self, mock_from_session, mock_write, mock_st, mock_vanna_service):
        """Test that a valid pair is generated, validated, and trained."""
        from utils.vanna_calls import auto_generate_sql_pairs

        mock_from_session.return_value = mock_vanna_service
        mock_st.progress.return_value = MagicMock()
        mock_st.empty.return_value = MagicMock()

        # LLM returns valid question+SQL on generation call
        mock_vanna_service.submit_prompt.side_effect = [
            "QUESTION: How many patients?\nSQL: SELECT COUNT(*) FROM patients;",
            "PASS",
        ]
        # SQL execution returns non-empty DataFrame
        mock_vanna_service.vn.run_sql.return_value = pd.DataFrame({"count": [42]})
        # check_references passes
        mock_vanna_service.check_references.return_value = "SELECT COUNT(*) FROM patients;"

        with patch("utils.security_validator.security_validator") as mock_sv:
            mock_sv.validate_sql_content.return_value = (True, [])
            results = auto_generate_sql_pairs(count=1)

        assert results["attempted"] == 1
        assert results["passed"] == 1
        assert results["failed"] == 0
        mock_write.assert_called_once()
        call_args = mock_write.call_args[0][0]
        assert call_args["question"] == "How many patients?"
        assert call_args["query"] == "SELECT COUNT(*) FROM patients;"

    @patch("utils.vanna_calls.st")
    @patch("utils.vanna_calls.write_to_file_and_training")
    @patch("utils.vanna_calls.VannaService.from_streamlit_session")
    def test_security_validation_blocks_forbidden_sql(self, mock_from_session, mock_write, mock_st, mock_vanna_service):
        """Test that pairs with forbidden references are rejected."""
        from utils.vanna_calls import auto_generate_sql_pairs

        mock_from_session.return_value = mock_vanna_service
        mock_st.progress.return_value = MagicMock()
        mock_st.empty.return_value = MagicMock()

        mock_vanna_service.submit_prompt.return_value = (
            "QUESTION: Get secret data\nSQL: SELECT * FROM secret_table;"
        )

        with patch("utils.security_validator.security_validator") as mock_sv:
            mock_sv.validate_sql_content.return_value = (False, ["Forbidden table reference: secret_table"])
            results = auto_generate_sql_pairs(count=1)

        assert results["passed"] == 0
        assert results["failed"] == 1
        assert "Security validation" in results["details"][0]["reason"]
        mock_write.assert_not_called()

    @patch("utils.vanna_calls.st")
    @patch("utils.vanna_calls.write_to_file_and_training")
    @patch("utils.vanna_calls.VannaService.from_streamlit_session")
    def test_sql_execution_error_rejected(self, mock_from_session, mock_write, mock_st, mock_vanna_service):
        """Test that pairs with SQL execution errors are rejected."""
        from utils.vanna_calls import auto_generate_sql_pairs

        mock_from_session.return_value = mock_vanna_service
        mock_st.progress.return_value = MagicMock()
        mock_st.empty.return_value = MagicMock()

        mock_vanna_service.submit_prompt.return_value = (
            "QUESTION: Bad query\nSQL: SELECT * FROM nonexistent;"
        )
        mock_vanna_service.check_references.return_value = "SELECT * FROM nonexistent;"
        mock_vanna_service.vn.run_sql.side_effect = Exception("relation does not exist")

        with patch("utils.security_validator.security_validator") as mock_sv:
            mock_sv.validate_sql_content.return_value = (True, [])
            results = auto_generate_sql_pairs(count=1)

        assert results["passed"] == 0
        assert results["failed"] == 1
        assert "SQL execution error" in results["details"][0]["reason"]
        mock_write.assert_not_called()

    @patch("utils.vanna_calls.st")
    @patch("utils.vanna_calls.write_to_file_and_training")
    @patch("utils.vanna_calls.VannaService.from_streamlit_session")
    def test_empty_results_rejected(self, mock_from_session, mock_write, mock_st, mock_vanna_service):
        """Test that pairs returning empty results are rejected."""
        from utils.vanna_calls import auto_generate_sql_pairs

        mock_from_session.return_value = mock_vanna_service
        mock_st.progress.return_value = MagicMock()
        mock_st.empty.return_value = MagicMock()

        mock_vanna_service.submit_prompt.return_value = (
            "QUESTION: Empty?\nSQL: SELECT * FROM patients WHERE 1=0;"
        )
        mock_vanna_service.check_references.return_value = "SELECT * FROM patients WHERE 1=0;"
        mock_vanna_service.vn.run_sql.return_value = pd.DataFrame()

        with patch("utils.security_validator.security_validator") as mock_sv:
            mock_sv.validate_sql_content.return_value = (True, [])
            results = auto_generate_sql_pairs(count=1)

        assert results["passed"] == 0
        assert results["failed"] == 1
        assert "empty results" in results["details"][0]["reason"]

    @patch("utils.vanna_calls.st")
    @patch("utils.vanna_calls.write_to_file_and_training")
    @patch("utils.vanna_calls.VannaService.from_streamlit_session")
    def test_semantic_review_fail_rejected(self, mock_from_session, mock_write, mock_st, mock_vanna_service):
        """Test that pairs failing semantic review are rejected."""
        from utils.vanna_calls import auto_generate_sql_pairs

        mock_from_session.return_value = mock_vanna_service
        mock_st.progress.return_value = MagicMock()
        mock_st.empty.return_value = MagicMock()

        # First submit_prompt = generation, second = review
        mock_vanna_service.submit_prompt.side_effect = [
            "QUESTION: How many?\nSQL: SELECT 1;",
            "FAIL - results do not answer the question",
        ]
        mock_vanna_service.check_references.return_value = "SELECT 1;"
        mock_vanna_service.vn.run_sql.return_value = pd.DataFrame({"col": [1]})

        with patch("utils.security_validator.security_validator") as mock_sv:
            mock_sv.validate_sql_content.return_value = (True, [])
            results = auto_generate_sql_pairs(count=1)

        assert results["passed"] == 0
        assert results["failed"] == 1
        assert "Semantic review failed" in results["details"][0]["reason"]

    @patch("utils.vanna_calls.st")
    @patch("utils.vanna_calls.write_to_file_and_training")
    @patch("utils.vanna_calls.VannaService.from_streamlit_session")
    def test_unparseable_response_rejected(self, mock_from_session, mock_write, mock_st, mock_vanna_service):
        """Test that unparseable LLM responses are handled gracefully."""
        from utils.vanna_calls import auto_generate_sql_pairs

        mock_from_session.return_value = mock_vanna_service
        mock_st.progress.return_value = MagicMock()
        mock_st.empty.return_value = MagicMock()

        mock_vanna_service.submit_prompt.return_value = "I cannot generate SQL for that."

        with patch("utils.security_validator.security_validator") as mock_sv:
            mock_sv.validate_sql_content.return_value = (True, [])
            results = auto_generate_sql_pairs(count=1)

        assert results["passed"] == 0
        assert results["failed"] == 1
        assert "parse" in results["details"][0]["reason"].lower()

    @patch("utils.vanna_calls.st")
    @patch("utils.vanna_calls.write_to_file_and_training")
    @patch("utils.vanna_calls.VannaService.from_streamlit_session")
    def test_check_references_blocks_forbidden(self, mock_from_session, mock_write, mock_st, mock_vanna_service):
        """Test that check_references returning None blocks the pair."""
        from utils.vanna_calls import auto_generate_sql_pairs

        mock_from_session.return_value = mock_vanna_service
        mock_st.progress.return_value = MagicMock()
        mock_st.empty.return_value = MagicMock()

        mock_vanna_service.submit_prompt.return_value = (
            "QUESTION: Get data\nSQL: SELECT * FROM forbidden_table;"
        )
        mock_vanna_service.check_references.return_value = None  # blocked

        with patch("utils.security_validator.security_validator") as mock_sv:
            mock_sv.validate_sql_content.return_value = (True, [])
            results = auto_generate_sql_pairs(count=1)

        assert results["passed"] == 0
        assert results["failed"] == 1
        assert "Forbidden reference" in results["details"][0]["reason"]

    @patch("utils.vanna_calls.st")
    @patch("utils.vanna_calls.write_to_file_and_training")
    @patch("utils.vanna_calls.VannaService.from_streamlit_session")
    def test_count_clamped_to_range(self, mock_from_session, mock_write, mock_st, mock_vanna_service):
        """Test that count is clamped between 1 and 50."""
        from utils.vanna_calls import auto_generate_sql_pairs

        mock_from_session.return_value = mock_vanna_service
        mock_st.progress.return_value = MagicMock()
        mock_st.empty.return_value = MagicMock()

        mock_vanna_service.submit_prompt.return_value = "garbage"

        with patch("utils.security_validator.security_validator") as mock_sv:
            mock_sv.validate_sql_content.return_value = (True, [])
            # count=0 should be clamped to 1
            results = auto_generate_sql_pairs(count=0)

        assert results["attempted"] == 1

    @patch("utils.vanna_calls.st")
    @patch("utils.vanna_calls.write_to_file_and_training")
    @patch("utils.vanna_calls.VannaService.from_streamlit_session")
    def test_multiple_pairs_mixed_results(self, mock_from_session, mock_write, mock_st, mock_vanna_service):
        """Test generating multiple pairs where some pass and some fail."""
        from utils.vanna_calls import auto_generate_sql_pairs

        mock_from_session.return_value = mock_vanna_service
        mock_st.progress.return_value = MagicMock()
        mock_st.empty.return_value = MagicMock()

        # Pair 1: succeeds, Pair 2: fails execution, Pair 3: succeeds
        mock_vanna_service.submit_prompt.side_effect = [
            "QUESTION: Q1\nSQL: SELECT 1;",
            "PASS",
            "QUESTION: Q2\nSQL: SELECT bad;",
            # no review call for pair 2 since it fails execution
            "QUESTION: Q3\nSQL: SELECT 3;",
            "PASS",
        ]
        mock_vanna_service.check_references.return_value = "sql"

        call_count = [0]

        def run_sql_side_effect(sql):
            call_count[0] += 1
            if call_count[0] == 2:
                raise Exception("syntax error")
            return pd.DataFrame({"v": [1]})

        mock_vanna_service.vn.run_sql.side_effect = run_sql_side_effect

        with patch("utils.security_validator.security_validator") as mock_sv:
            mock_sv.validate_sql_content.return_value = (True, [])
            results = auto_generate_sql_pairs(count=3)

        assert results["attempted"] == 3
        assert results["passed"] == 2
        assert results["failed"] == 1

    @patch("utils.vanna_calls.st")
    @patch("utils.vanna_calls.VannaService.from_streamlit_session")
    def test_vanna_service_unavailable(self, mock_from_session, mock_st):
        """Test graceful handling when VannaService is unavailable."""
        from utils.vanna_calls import auto_generate_sql_pairs

        mock_from_session.return_value = None

        results = auto_generate_sql_pairs(count=1)

        assert results["attempted"] == 0
        assert "error" in results

    @patch("utils.vanna_calls.st")
    @patch("utils.vanna_calls.write_to_file_and_training")
    @patch("utils.vanna_calls.VannaService.from_streamlit_session")
    def test_llm_returns_exception(self, mock_from_session, mock_write, mock_st, mock_vanna_service):
        """Test handling when submit_prompt returns an Exception object."""
        from utils.vanna_calls import auto_generate_sql_pairs

        mock_from_session.return_value = mock_vanna_service
        mock_st.progress.return_value = MagicMock()
        mock_st.empty.return_value = MagicMock()

        mock_vanna_service.submit_prompt.return_value = Exception("API error")

        with patch("utils.security_validator.security_validator") as mock_sv:
            mock_sv.validate_sql_content.return_value = (True, [])
            results = auto_generate_sql_pairs(count=1)

        assert results["passed"] == 0
        assert results["failed"] == 1
        assert "LLM generation failed" in results["details"][0]["reason"]
