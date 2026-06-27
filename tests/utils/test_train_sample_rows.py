"""Tests for train_sample_rows() de-identified sample row generation."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MOCK_SECRETS = {
    "ai_keys": {"ollama_model": "test_model"},
    "rag_model": {"chroma_path": "/tmp/test_chroma"},
    "postgres": {
        "host": "localhost",
        "port": 5432,
        "database": "test_db",
        "user": "test_user",
        "password": "test_pass",
    },
    "security": {"allow_llm_to_see_data": True, "train_sample_rows": True},
}


def _make_secrets(overrides: dict | None = None) -> dict:
    """Return a copy of MOCK_SECRETS with optional overrides merged in."""
    import copy

    s = copy.deepcopy(MOCK_SECRETS)
    if overrides:
        for section, values in overrides.items():
            if isinstance(values, dict):
                s.setdefault(section, {}).update(values)
            else:
                s[section] = values
    return s


SAMPLE_LLM_MARKDOWN = (
    "| id | first_name | last_name | dob        |\n"
    "|----|------------|-----------|------------|\n"
    "| 1  | Jane       | Smith     | 1990-01-15 |\n"
    "| 2  | Bob        | Jones     | 1985-06-22 |\n"
    "| 3  | Alice      | Brown     | 1978-11-03 |\n"
)


# ---------------------------------------------------------------------------
# _parse_markdown_table tests
# ---------------------------------------------------------------------------


class TestParseMarkdownTable:
    """Tests for the markdown table parser."""

    def test_valid_table(self):
        from utils.vanna_calls import _parse_markdown_table

        result = _parse_markdown_table(SAMPLE_LLM_MARKDOWN)
        assert result is not None
        assert "| id |" in result
        assert "Jane" in result

    def test_table_with_surrounding_text(self):
        from utils.vanna_calls import _parse_markdown_table

        text = "Here are the de-identified rows:\n\n" + SAMPLE_LLM_MARKDOWN + "\nDone."
        result = _parse_markdown_table(text)
        assert result is not None
        assert "Jane" in result
        # Surrounding text should NOT be included
        assert "Here are" not in result

    def test_pipe_in_prose_before_table_still_parses(self):
        from utils.vanna_calls import _parse_markdown_table

        text = "The format is col1 | col2 for reference.\n\n" + SAMPLE_LLM_MARKDOWN
        result = _parse_markdown_table(text)
        assert result is not None
        assert "Jane" in result
        assert "col1 | col2 for reference" not in result

    def test_no_table_returns_none(self):
        from utils.vanna_calls import _parse_markdown_table

        result = _parse_markdown_table("No table here, just text.")
        assert result is None

    def test_header_only_returns_none(self):
        from utils.vanna_calls import _parse_markdown_table

        text = "| col1 | col2 |\n|------|------|\n"
        result = _parse_markdown_table(text)
        # Only 2 lines — needs at least 3 (header + separator + 1 data row)
        assert result is None

    def test_empty_string_returns_none(self):
        from utils.vanna_calls import _parse_markdown_table

        result = _parse_markdown_table("")
        assert result is None


# ---------------------------------------------------------------------------
# _build_deidentification_prompt tests
# ---------------------------------------------------------------------------


class TestBuildDeidentificationPrompt:
    """Tests for the de-identification prompt builder."""

    def test_includes_table_name_and_ddl(self):
        from utils.vanna_calls import _build_deidentification_prompt

        df = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})
        prompt = _build_deidentification_prompt("public", "patients", "CREATE TABLE ...", df)
        assert "public.patients" in prompt
        assert "CREATE TABLE" in prompt
        assert "Alice" in prompt
        assert "Bob" in prompt


# ---------------------------------------------------------------------------
# train_sample_rows integration tests (mocked DB + LLM)
# ---------------------------------------------------------------------------


class TestTrainSampleRows:
    """Integration tests for train_sample_rows with mocked dependencies."""

    @pytest.fixture(autouse=True)
    def _patch_streamlit(self):
        """Patch Streamlit UI calls that are irrelevant to logic testing."""
        with (
            patch("utils.vanna_calls.st.toast"),
            patch("utils.vanna_calls.st.progress", return_value=MagicMock()),
            patch("utils.vanna_calls.st.empty", return_value=MagicMock()),
            patch("utils.vanna_calls.st.success"),
            patch("utils.vanna_calls.st.warning"),
            patch("utils.vanna_calls.st.error"),
            patch("utils.vanna_calls.st.status", return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock())),
        ):
            yield

    def _setup_mocks(self, secrets_overrides=None, tables=None, sample_df=None, llm_response=None):
        """Create a standard set of mocks for train_sample_rows tests.

        Returns a dict of mock objects for further assertions.
        """
        secrets = _make_secrets(secrets_overrides)
        if tables is None:
            tables = [("patients",), ("encounters",)]
        if sample_df is None:
            sample_df = pd.DataFrame({"id": [1, 2], "name": ["Real Name", "Real Name 2"]})
        if llm_response is None:
            llm_response = SAMPLE_LLM_MARKDOWN

        mock_vanna_service = MagicMock()
        mock_vanna_service.get_training_data.return_value = pd.DataFrame()
        mock_vanna_service.submit_prompt.return_value = llm_response
        mock_vanna_service.train.return_value = True

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = tables

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        return {
            "secrets": secrets,
            "vanna_service": mock_vanna_service,
            "conn": mock_conn,
            "cursor": mock_cursor,
            "sample_df": sample_df,
        }

    def test_calls_llm_for_deidentification(self):
        """Verify that the LLM is called to de-identify rows, not stored raw."""
        from utils.vanna_calls import train_sample_rows

        mocks = self._setup_mocks()
        sample_df = mocks["sample_df"]

        with (
            patch("utils.vanna_calls.st.secrets", new=mocks["secrets"]),
            patch("utils.vanna_calls.VannaService.from_streamlit_session", return_value=mocks["vanna_service"]),
            patch("utils.vanna_calls.psycopg2.connect", return_value=mocks["conn"]),
            patch("utils.vanna_calls._get_random_sample_data", return_value=sample_df),
            patch("utils.vanna_calls._get_single_table_ddl", return_value="CREATE TABLE patients (...)"),
            patch("utils.vanna_calls.read_forbidden_from_json", return_value=([], [], "")),
        ):
            result = train_sample_rows(clear_existing=False)

        assert result is True
        # LLM should have been called once per table
        assert mocks["vanna_service"].submit_prompt.call_count == 2

    def test_forbidden_tables_are_skipped(self):
        """Verify forbidden tables are excluded from the table list query."""
        from utils.vanna_calls import train_sample_rows

        # Only non-forbidden table returned by DB
        mocks = self._setup_mocks(tables=[("encounters",)])

        with (
            patch("utils.vanna_calls.st.secrets", new=mocks["secrets"]),
            patch("utils.vanna_calls.VannaService.from_streamlit_session", return_value=mocks["vanna_service"]),
            patch("utils.vanna_calls.psycopg2.connect", return_value=mocks["conn"]),
            patch("utils.vanna_calls._get_random_sample_data", return_value=mocks["sample_df"]),
            patch("utils.vanna_calls._get_single_table_ddl", return_value="CREATE TABLE ..."),
            patch(
                "utils.vanna_calls.read_forbidden_from_json",
                return_value=(["forbidden_table"], [], "'forbidden_table'"),
            ),
        ):
            result = train_sample_rows(clear_existing=False)

        assert result is True
        # The SQL query should have been called with forbidden table in params
        execute_call = mocks["cursor"].execute.call_args
        # The params list should include the forbidden table name
        assert "forbidden_table" in execute_call[0][1]

    def test_config_toggle_false_skips_step(self):
        """Verify that train_sample_rows=False in config skips the entire step."""
        from utils.vanna_calls import train_sample_rows

        secrets = _make_secrets({"security": {"train_sample_rows": False}})

        with patch("utils.vanna_calls.st.secrets", new=secrets):
            result = train_sample_rows()

        assert result is False

    def test_allow_llm_to_see_data_false_returns_false_without_llm_call(self):
        """Verify that allow_llm_to_see_data=False blocks execution and never calls submit_prompt."""
        from utils.vanna_calls import train_sample_rows

        secrets = _make_secrets({"security": {"allow_llm_to_see_data": False, "train_sample_rows": True}})
        mocks = self._setup_mocks()

        with (
            patch("utils.vanna_calls.st.secrets", new=secrets),
            patch("utils.vanna_calls.VannaService.from_streamlit_session", return_value=mocks["vanna_service"]),
        ):
            result = train_sample_rows()

        assert result is False
        mocks["vanna_service"].submit_prompt.assert_not_called()

    def test_old_entries_cleared_before_new_ones(self):
        """Verify that existing sample_rows entries are removed when clear_existing=True."""
        from utils.vanna_calls import train_sample_rows

        mocks = self._setup_mocks()

        # Simulate existing training data with sample_rows entries
        existing_data = pd.DataFrame(
            {
                "id": ["sample_rows_public_old_table", "other_doc_id"],
                "content": [
                    "SAMPLE DATA (synthetic/de-identified) for public.old_table:\n| col |",
                    "Some other documentation",
                ],
                "training_data_type": ["documentation", "documentation"],
            }
        )
        mocks["vanna_service"].get_training_data.return_value = existing_data

        with (
            patch("utils.vanna_calls.st.secrets", new=mocks["secrets"]),
            patch("utils.vanna_calls.VannaService.from_streamlit_session", return_value=mocks["vanna_service"]),
            patch("utils.vanna_calls.psycopg2.connect", return_value=mocks["conn"]),
            patch("utils.vanna_calls._get_random_sample_data", return_value=mocks["sample_df"]),
            patch("utils.vanna_calls._get_single_table_ddl", return_value="CREATE TABLE ..."),
            patch("utils.vanna_calls.read_forbidden_from_json", return_value=([], [], "")),
        ):
            train_sample_rows(clear_existing=True)

        # Should have removed the old sample_rows entry but not the other doc
        remove_calls = mocks["vanna_service"].remove_from_training.call_args_list
        removed_ids = [c[0][0] for c in remove_calls]
        assert "sample_rows_public_old_table" in removed_ids
        assert "other_doc_id" not in removed_ids

    def test_empty_tables_handled_gracefully(self):
        """Verify that tables with no rows are skipped without error."""
        from utils.vanna_calls import train_sample_rows

        mocks = self._setup_mocks(tables=[("empty_table",)])
        empty_df = pd.DataFrame()

        with (
            patch("utils.vanna_calls.st.secrets", new=mocks["secrets"]),
            patch("utils.vanna_calls.VannaService.from_streamlit_session", return_value=mocks["vanna_service"]),
            patch("utils.vanna_calls.psycopg2.connect", return_value=mocks["conn"]),
            patch("utils.vanna_calls._get_random_sample_data", return_value=empty_df),
            patch("utils.vanna_calls._get_single_table_ddl", return_value="CREATE TABLE ..."),
            patch("utils.vanna_calls.read_forbidden_from_json", return_value=([], [], "")),
        ):
            result = train_sample_rows(clear_existing=False)

        # No tables processed but function completes without error
        assert result is False
        # LLM should never have been called (no data to de-identify)
        mocks["vanna_service"].submit_prompt.assert_not_called()

    def test_raw_data_never_passed_to_train(self):
        """Verify that the original raw data values do not appear in train() calls."""
        from utils.vanna_calls import train_sample_rows

        raw_df = pd.DataFrame({"id": [999], "patient_name": ["REAL_SECRET_NAME"], "ssn": ["123-45-6789"]})
        mocks = self._setup_mocks(tables=[("patients",)], sample_df=raw_df)

        with (
            patch("utils.vanna_calls.st.secrets", new=mocks["secrets"]),
            patch("utils.vanna_calls.VannaService.from_streamlit_session", return_value=mocks["vanna_service"]),
            patch("utils.vanna_calls.psycopg2.connect", return_value=mocks["conn"]),
            patch("utils.vanna_calls._get_random_sample_data", return_value=raw_df),
            patch("utils.vanna_calls._get_single_table_ddl", return_value="CREATE TABLE patients (...)"),
            patch("utils.vanna_calls.read_forbidden_from_json", return_value=([], [], "")),
        ):
            train_sample_rows(clear_existing=False)

        # Inspect what was passed to train()
        for train_call in mocks["vanna_service"].train.call_args_list:
            doc_text = train_call[1].get("documentation", "")
            # Raw PHI values must NOT appear in the stored documentation
            assert "REAL_SECRET_NAME" not in doc_text
            assert "123-45-6789" not in doc_text
            # Instead, the LLM-generated synthetic data should be present
            assert "SAMPLE DATA (synthetic/de-identified)" in doc_text

    def test_llm_error_response_skips_table(self):
        """Verify that an LLM error response causes the table to be skipped."""
        from utils.vanna_calls import train_sample_rows

        mocks = self._setup_mocks(tables=[("patients",)])
        mocks["vanna_service"].submit_prompt.return_value = Exception("LLM error")

        with (
            patch("utils.vanna_calls.st.secrets", new=mocks["secrets"]),
            patch("utils.vanna_calls.VannaService.from_streamlit_session", return_value=mocks["vanna_service"]),
            patch("utils.vanna_calls.psycopg2.connect", return_value=mocks["conn"]),
            patch("utils.vanna_calls._get_random_sample_data", return_value=mocks["sample_df"]),
            patch("utils.vanna_calls._get_single_table_ddl", return_value="CREATE TABLE ..."),
            patch("utils.vanna_calls.read_forbidden_from_json", return_value=([], [], "")),
        ):
            result = train_sample_rows(clear_existing=False)

        assert result is False
        mocks["vanna_service"].train.assert_not_called()

    def test_unparseable_llm_response_skips_table(self):
        """Verify that a non-table LLM response causes the table to be skipped."""
        from utils.vanna_calls import train_sample_rows

        mocks = self._setup_mocks(tables=[("patients",)])
        mocks["vanna_service"].submit_prompt.return_value = "Sorry, I can't do that."

        with (
            patch("utils.vanna_calls.st.secrets", new=mocks["secrets"]),
            patch("utils.vanna_calls.VannaService.from_streamlit_session", return_value=mocks["vanna_service"]),
            patch("utils.vanna_calls.psycopg2.connect", return_value=mocks["conn"]),
            patch("utils.vanna_calls._get_random_sample_data", return_value=mocks["sample_df"]),
            patch("utils.vanna_calls._get_single_table_ddl", return_value="CREATE TABLE ..."),
            patch("utils.vanna_calls.read_forbidden_from_json", return_value=([], [], "")),
        ):
            result = train_sample_rows(clear_existing=False)

        assert result is False
        mocks["vanna_service"].train.assert_not_called()

    def test_metadata_includes_auto_type_and_table_name(self):
        """Verify stored entries have correct metadata tags."""
        from utils.vanna_calls import train_sample_rows

        mocks = self._setup_mocks(tables=[("patients",)])

        with (
            patch("utils.vanna_calls.st.secrets", new=mocks["secrets"]),
            patch("utils.vanna_calls.VannaService.from_streamlit_session", return_value=mocks["vanna_service"]),
            patch("utils.vanna_calls.psycopg2.connect", return_value=mocks["conn"]),
            patch("utils.vanna_calls._get_random_sample_data", return_value=mocks["sample_df"]),
            patch("utils.vanna_calls._get_single_table_ddl", return_value="CREATE TABLE patients (...)"),
            patch("utils.vanna_calls.read_forbidden_from_json", return_value=([], [], "")),
        ):
            train_sample_rows(clear_existing=False)

        train_call = mocks["vanna_service"].train.call_args
        metadata = train_call[1]["metadata"]
        assert metadata["auto_type"] == "sample_rows"
        assert metadata["table_name"] == "patients"
        assert train_call[1]["id"] == "sample_rows_public_patients"

    def test_config_toggle_default_true(self):
        """Verify that when train_sample_rows is not in config, it defaults to True (runs)."""
        from utils.vanna_calls import train_sample_rows

        # Remove train_sample_rows from security config
        secrets = _make_secrets()
        del secrets["security"]["train_sample_rows"]

        mocks = self._setup_mocks()

        with (
            patch("utils.vanna_calls.st.secrets", new=secrets),
            patch("utils.vanna_calls.VannaService.from_streamlit_session", return_value=mocks["vanna_service"]),
            patch("utils.vanna_calls.psycopg2.connect", return_value=mocks["conn"]),
            patch("utils.vanna_calls._get_random_sample_data", return_value=mocks["sample_df"]),
            patch("utils.vanna_calls._get_single_table_ddl", return_value="CREATE TABLE ..."),
            patch("utils.vanna_calls.read_forbidden_from_json", return_value=([], [], "")),
        ):
            result = train_sample_rows(clear_existing=False)

        # Should proceed (not skip)
        assert result is True
