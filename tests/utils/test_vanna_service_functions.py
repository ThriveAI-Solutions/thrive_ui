import json
from unittest.mock import MagicMock, mock_open, patch

import pandas as pd
import pytest

from utils.vanna_calls import (
    VannaService,
    generate_plot_cached,
    generate_plotly_code_cached,
    generate_summary_cached,
    is_sql_valid_cached,
    remove_from_file_training,
    should_generate_chart_cached,
    training_plan,
)


# Mock the streamlit secrets for testing
@pytest.fixture
def mock_streamlit_secrets():
    with patch(
        "streamlit.secrets",
        new={
            "ai_keys": {
                "ollama_model": "llama3",
                "vanna_api": "mock_vanna_api",
                "vanna_model": "mock_vanna_model",
                "anthropic_api": "mock_anthropic_api",
                "anthropic_model": "claude-3-sonnet-20240229",
            },
            "rag_model": {"chroma_path": "./chromadb"},
            "postgres": {
                "host": "localhost",
                "port": 5432,
                "database": "thrive",
                "user": "postgres",
                "password": "postgres",
            },
            "security": {"allow_llm_to_see_data": False},
        },
    ):
        yield


# Test for additional VannaService methods
@pytest.mark.usefixtures("mock_streamlit_secrets")
class TestVannaServiceAdditional:
    @patch("utils.vanna_calls.VannaService._setup_vanna")
    def test_is_sql_valid(self, mock_setup_vanna):
        service = VannaService()
        service.vn = MagicMock()
        service.vn.is_sql_valid.return_value = True

        result = service.is_sql_valid("SELECT * FROM test_table")

        assert result is True
        service.vn.is_sql_valid.assert_called_once_with(sql="SELECT * FROM test_table")

    @patch("utils.vanna_calls.VannaService._setup_vanna")
    def test_run_sql(self, mock_setup_vanna):
        service = VannaService()
        service.vn = MagicMock()
        mock_df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
        service.vn.run_sql.return_value = mock_df

        result = service.run_sql("SELECT * FROM test_table")

        assert result.equals(mock_df)
        service.vn.run_sql.assert_called_once_with(sql="SELECT * FROM test_table")

    @patch("utils.vanna_calls.VannaService._setup_vanna")
    def test_should_generate_chart(self, mock_setup_vanna):
        service = VannaService()
        service.vn = MagicMock()
        service.vn.should_generate_chart.return_value = True

        mock_df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
        result = service.should_generate_chart("Test question", "SELECT * FROM test_table", mock_df)

        assert result is True
        service.vn.should_generate_chart.assert_called_once_with(df=mock_df)

    @patch("utils.vanna_calls.VannaService._setup_vanna")
    def test_generate_plotly_code(self, mock_setup_vanna):
        service = VannaService()
        service.vn = MagicMock()
        service.vn.generate_plotly_code.return_value = "mock_plotly_code"

        mock_df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
        result, elapsed_time = service.generate_plotly_code("Test question", "SELECT * FROM test_table", mock_df)

        assert result == "mock_plotly_code"
        assert isinstance(elapsed_time, float)
        service.vn.generate_plotly_code.assert_called_once_with(
            question="Test question", sql="SELECT * FROM test_table", df=mock_df
        )

    @patch("utils.vanna_calls.VannaService._setup_vanna")
    def test_generate_plot(self, mock_setup_vanna):
        service = VannaService()
        service.vn = MagicMock()
        service.vn.get_plotly_figure.return_value = "mock_plotly_figure"

        mock_df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
        result, elapsed_time = service.generate_plot("mock_code", mock_df)

        assert result == "mock_plotly_figure"
        assert isinstance(elapsed_time, float)
        service.vn.get_plotly_figure.assert_called_once_with(plotly_code="mock_code", df=mock_df)

    @patch("utils.vanna_calls.VannaService._setup_vanna")
    def test_generate_followup_questions(self, mock_setup_vanna):
        service = VannaService()
        service.vn = MagicMock()
        service.vn.generate_followup_questions.return_value = ["Question 1", "Question 2"]

        mock_df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
        result = service.generate_followup_questions("Test question", "SELECT * FROM test_table", mock_df)

        assert result == ["Question 1", "Question 2"]
        service.vn.generate_followup_questions.assert_called_once_with(
            question="Test question", sql="SELECT * FROM test_table", df=mock_df
        )

    @patch("utils.vanna_calls.VannaService._setup_vanna")
    def test_generate_summary(self, mock_setup_vanna):
        service = VannaService()
        service.vn = MagicMock()
        service.vn.generate_summary.return_value = "This is a summary"

        mock_df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
        result, elapsed_time = service.generate_summary("Test question", mock_df)

        assert result == "This is a summary"
        assert isinstance(elapsed_time, float)
        service.vn.generate_summary.assert_called_once_with(question="Test question", df=mock_df)


# Test for cached functions continued
@pytest.mark.usefixtures("mock_streamlit_secrets")
class TestMoreCachedFunctions:
    @patch("utils.vanna_calls.VannaService.get_instance")
    def test_is_sql_valid_cached(self, mock_get_instance):
        mock_service = MagicMock()
        mock_get_instance.return_value = mock_service
        mock_service.is_sql_valid.return_value = True

        result = is_sql_valid_cached("SELECT * FROM test_table")

        assert result is True
        mock_service.is_sql_valid.assert_called_once_with("SELECT * FROM test_table")

    @patch("utils.vanna_calls.VannaService.get_instance")
    def test_should_generate_chart_cached(self, mock_get_instance):
        mock_service = MagicMock()
        mock_get_instance.return_value = mock_service
        mock_service.should_generate_chart.return_value = True

        mock_df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
        result = should_generate_chart_cached("Test question", "SELECT * FROM test_table", mock_df)

        assert result is True
        mock_service.should_generate_chart.assert_called_once_with("Test question", "SELECT * FROM test_table", mock_df)

    @patch("utils.vanna_calls.VannaService.get_instance")
    def test_generate_plotly_code_cached(self, mock_get_instance):
        mock_service = MagicMock()
        mock_get_instance.return_value = mock_service
        mock_service.generate_plotly_code.return_value = ("mock_plotly_code", 0.5)

        mock_df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
        code, elapsed_time = generate_plotly_code_cached("Test question", "SELECT * FROM test_table", mock_df)

        assert code == "mock_plotly_code"
        assert elapsed_time == 0.5
        mock_service.generate_plotly_code.assert_called_once_with("Test question", "SELECT * FROM test_table", mock_df)

    @patch("utils.vanna_calls.VannaService.get_instance")
    def test_generate_plot_cached(self, mock_get_instance):
        mock_service = MagicMock()
        mock_get_instance.return_value = mock_service
        mock_service.generate_plot.return_value = ("mock_plot", 0.5)

        mock_df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
        plot, elapsed_time = generate_plot_cached("mock_code", mock_df)

        assert plot == "mock_plot"
        assert elapsed_time == 0.5
        mock_service.generate_plot.assert_called_once_with("mock_code", mock_df)

    @patch("utils.vanna_calls.VannaService.get_instance")
    def test_generate_summary_cached(self, mock_get_instance):
        mock_service = MagicMock()
        mock_get_instance.return_value = mock_service
        mock_service.generate_summary.return_value = ("This is a summary", 0.5)

        mock_df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
        summary, elapsed_time = generate_summary_cached("Test question", mock_df)

        assert summary == "This is a summary"
        assert elapsed_time == 0.5
        mock_service.generate_summary.assert_called_once_with("Test question", mock_df)


# Test for file training functions
@pytest.mark.usefixtures("mock_streamlit_secrets")
class TestFileTrainingFunctions:
    @patch("utils.vanna_calls.VannaService.get_instance")
    def test_remove_from_file_training(self, mock_get_instance):
        mock_service = MagicMock()
        mock_get_instance.return_value = mock_service

        # Mock the training data returned by get_training_data
        mock_training_df = pd.DataFrame(
            {"id": [1], "question": ["How many users are there?"], "sql": ["SELECT COUNT(*) FROM users"]}
        )
        mock_service.get_training_data.return_value = mock_training_df

        # Mock the training data JSON file
        training_data = {
            "sample_queries": [{"question": "How many users are there?", "query": "SELECT COUNT(*) FROM users"}]
        }

        # Use mock_open to avoid creating real files
        m = mock_open(read_data=json.dumps(training_data))

        with patch("builtins.open", m), patch("pathlib.Path.open", m), patch("json.load", return_value=training_data):
            # Test the function
            new_entry = {"question": "How many users are there?"}
            remove_from_file_training(new_entry)

            # Verify remove_from_training was called
            mock_service.remove_from_training.assert_called_once_with(1)

            # Check that json.dump was called with the updated data
            # This verifies the write operation would have happened correctly
            args, kwargs = m.return_value.__enter__.return_value.write.call_args
            written_data = args[0]
            # After removal, the query shouldn't be in the data anymore
            assert "How many users are there?" not in written_data

    @patch("utils.vanna_calls.VannaService.get_instance")
    @patch("utils.vanna_calls.read_forbidden_from_json")
    def test_training_plan(self, mock_read_forbidden, mock_get_instance):
        # Create a mock service
        mock_service = MagicMock()
        mock_get_instance.return_value = mock_service

        # Mock the forbidden tables
        mock_read_forbidden.return_value = (["forbidden_table"], ["password"], "'forbidden_table'")

        # Mock the information schema query result
        mock_df = pd.DataFrame(
            {
                "table_schema": ["public", "public"],
                "table_name": ["users", "products"],
                "column_name": ["id", "name"],
                "data_type": ["integer", "varchar"],
                "is_nullable": ["NO", "YES"],
            }
        )
        mock_service.run_sql.return_value = mock_df

        # Mock the training plan
        mock_service.get_training_plan_generic.return_value = "training_plan"

        # Call the function
        training_plan()

        # Verify run_sql was called with the correct SQL
        mock_service.run_sql.assert_called_once()
        sql_arg = mock_service.run_sql.call_args[0][0]
        assert "SELECT" in sql_arg
        assert "information_schema.columns" in sql_arg
        assert "table_schema = 'public'" in sql_arg
        assert "table_name NOT IN ('forbidden_table')" in sql_arg

        # Verify get_training_plan_generic was called with the correct dataframe
        mock_service.get_training_plan_generic.assert_called_once_with(mock_df)

        # Verify train was called with the correct plan
        mock_service.train.assert_called_once_with(plan="training_plan")
