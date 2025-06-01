from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pandas as pd
import pytest
import streamlit as st
from pandas import DataFrame

from utils.vanna_calls import (
    MyVannaAnthropic,
    MyVannaAnthropicChromaDB,
    MyVannaOllama,
    MyVannaOllamaChromaDB,
    VannaService,
    check_references,
    read_forbidden_from_json,
    remove_from_file_training,
    train_ddl,
    train_file,
    training_plan,
    write_to_file_and_training,
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
            "security": {"allow_llm_to_see_data": True},
        },
    ):
        yield


@pytest.mark.usefixtures("mock_streamlit_secrets")
class TestVannaServiceExceptions:
    def test_init_exception_anthropic(self):
        """Test that exceptions during VannaAnthropic initialization are handled properly."""
        with patch(
            "utils.vanna_calls.VannaDB_VectorStore.__init__", side_effect=Exception("VannaDB initialization error")
        ):
            # Should catch the exception and log it
            with pytest.raises(Exception, match="VannaDB initialization error"):
                instance = MyVannaAnthropic()
            # Further assertions can be made on mock_logger if needed

    def test_init_exception_anthropic_chromadb(self):
        """Test that exceptions during VannaAnthropicChromaDB initialization are handled properly."""
        with patch(
            "utils.chromadb_vector.ChromaDB_VectorStore.__init__", side_effect=Exception("ChromaDB initialization error")
        ):
            # Should catch the exception and log it
            with pytest.raises(Exception, match="ChromaDB initialization error"):
                instance = MyVannaAnthropicChromaDB(user_role=0)
            # Further assertions can be made on mock_logger if needed

    def test_init_exception_ollama(self):
        """Test that exceptions during VannaOllama initialization are handled properly."""
        with patch(
            "utils.vanna_calls.VannaDB_VectorStore.__init__", side_effect=Exception("VannaDB initialization error")
        ):
            # Should catch the exception and log it
            with pytest.raises(Exception, match="VannaDB initialization error"):
                instance = MyVannaOllama()
            # Further assertions can be made on mock_logger if needed

    def test_init_exception_ollama_chromadb(self):
        """Test that exceptions during VannaOllamaChromaDB initialization are handled properly."""
        with patch(
            "utils.chromadb_vector.ChromaDB_VectorStore.__init__", side_effect=Exception("ChromaDB initialization error")
        ):
            # Should catch the exception and log it
            with pytest.raises(Exception, match="ChromaDB initialization error"):
                instance = MyVannaOllamaChromaDB(user_role=0)
            # Further assertions can be made on mock_logger if needed

    @patch("streamlit.error")
    def test_setup_vanna_exception(self, mock_st_error):
        """Test that exceptions during VannaService._setup_vanna are caught, logged, but then re-raised."""
        VannaService._instance = None
        service = VannaService() # self.vn is None, self.user_role is 0

        # service.vn is already None from __init__
        # Force a configuration that will cause an exception in _setup_vanna
        with patch.dict(st.secrets, {"ai_keys": {}}): # Makes st.secrets.ai_keys an empty dict
            with pytest.raises(KeyError): # Expecting a KeyError from accessing st.secrets["ai_keys"]["vanna_api"]
                service._setup_vanna(user_role=0) # Pass a user_role as the method expects

        # Verify the error was logged via streamlit
        mock_st_error.assert_called_once()
        assert "Error setting up Vanna:" in mock_st_error.call_args[0][0]

    @pytest.mark.skip(reason="Streamlit caching interferes with this test")
    @patch("streamlit.error")
    def test_generate_questions_exception(self, mock_st_error):
        """Test that exceptions in generate_questions are handled properly."""
        with patch.object(VannaService, "_setup_vanna"):
            # Create a fresh instance of VannaService with a mocked vn
            service = VannaService()
            service.vn = MagicMock()
            service.vn.generate_questions.side_effect = Exception("Failed to generate questions")

            # Cache the service.generate_questions method with a clear_func parameter
            # This allows us to directly manipulate the function's cache
            original_method = service.generate_questions

            # We need to directly test without caching, so patch the method to call through to the wrapped function
            with patch.object(
                VannaService, "generate_questions", side_effect=lambda: original_method.__wrapped__(service)
            ):
                result = service.generate_questions()

                # Verify the exception was logged via streamlit
                mock_st_error.assert_called_once()
                # Verify the error message contains expected text
                assert "Error generating questions" in mock_st_error.call_args[0][0]
                # Verify the method returns None on exception
                assert result is None

    @patch("streamlit.error")
    def test_generate_sql_exception(self, mock_st_error):
        """Test that exceptions in generate_sql are handled properly."""
        with patch.object(VannaService, "_setup_vanna"):
            service = VannaService()
            service.vn = MagicMock()
            service.vn.generate_sql.side_effect = Exception("Failed to generate SQL")

            with patch.dict("streamlit.secrets", {"security": {"allow_llm_to_see_data": False}}):
                result, elapsed_time = service.generate_sql("Show me the data")

                # Should return None and display error
                assert result is None
                assert elapsed_time == 0
                mock_st_error.assert_called_once()
                assert "Error generating SQL" in mock_st_error.call_args[0][0]

    @patch("streamlit.error")
    def test_is_sql_valid_exception(self, mock_st_error):
        """Test that exceptions in is_sql_valid are handled properly."""
        with patch.object(VannaService, "_setup_vanna"):
            service = VannaService()
            service.vn = MagicMock()
            service.vn.is_sql_valid.side_effect = Exception("SQL validation error")

            result = service.is_sql_valid("SELECT * FROM table")

            # Should return False and display error
            assert result is False
            mock_st_error.assert_called_once()
            assert "Error checking SQL validity" in mock_st_error.call_args[0][0]

    @patch("streamlit.error")
    def test_run_sql_exception(self, mock_st_error):
        """Test that exceptions in run_sql are handled properly."""
        with patch.object(VannaService, "_setup_vanna"):
            service = VannaService()
            service.vn = MagicMock()
            service.vn.run_sql.side_effect = Exception("Failed to run SQL")

            result = service.run_sql("SELECT * FROM table")

            # Should return None and display error
            assert result is None
            mock_st_error.assert_called_once()
            assert "Error running SQL" in mock_st_error.call_args[0][0]

    @patch("streamlit.error")
    def test_should_generate_chart_exception(self, mock_st_error):
        """Test that exceptions in should_generate_chart are handled properly."""
        with patch.object(VannaService, "_setup_vanna"):
            service = VannaService()
            service.vn = MagicMock()
            service.vn.should_generate_chart.side_effect = Exception("Chart generation decision error")

            result = service.should_generate_chart("question", "sql", DataFrame())

            # Should return False and display error
            assert result is False
            mock_st_error.assert_called_once()
            assert "Error checking if we should generate a chart" in mock_st_error.call_args[0][0]

    @patch("streamlit.error")
    def test_generate_plotly_code_exception(self, mock_st_error):
        """Test that exceptions in generate_plotly_code are handled properly."""
        with patch.object(VannaService, "_setup_vanna"):
            service = VannaService()
            service.vn = MagicMock()
            service.vn.generate_plotly_code.side_effect = Exception("Failed to generate Plotly code")

            result, elapsed_time = service.generate_plotly_code("question", "sql", DataFrame())

            # Should return None and display error
            assert result is None
            assert elapsed_time == 0
            mock_st_error.assert_called_once()
            assert "Error generating Plotly code" in mock_st_error.call_args[0][0]

    @patch("streamlit.error")
    def test_generate_plot_exception(self, mock_st_error):
        """Test that exceptions in generate_plot are handled properly."""
        with patch.object(VannaService, "_setup_vanna"):
            service = VannaService()
            service.vn = MagicMock()
            service.vn.get_plotly_figure.side_effect = Exception("Failed to generate Plotly figure")

            result, elapsed_time = service.generate_plot("code", DataFrame())

            # Should return None and display error
            assert result is None
            assert elapsed_time == 0
            mock_st_error.assert_called_once()
            assert "Error generating Plotly chart" in mock_st_error.call_args[0][0]

    @patch("streamlit.error")
    def test_generate_followup_questions_exception(self, mock_st_error):
        """Test that exceptions in generate_followup_questions are handled properly."""
        with patch.object(VannaService, "_setup_vanna"):
            service = VannaService()
            service.vn = MagicMock()
            service.vn.generate_followup_questions.side_effect = Exception("Failed to generate followup questions")

            result = service.generate_followup_questions("question", "sql", DataFrame())

            # Should return empty list and display error
            assert result == []
            mock_st_error.assert_called_once()
            assert "Error generating followup questions" in mock_st_error.call_args[0][0]

    @pytest.mark.skip(reason="Streamlit caching interferes with this test")
    def test_generate_summary_exception(self):
        """Test that exceptions in generate_summary are handled properly."""
        with patch.object(VannaService, "_setup_vanna"):
            service = VannaService()
            service.vn = MagicMock()
            service.vn.generate_summary.side_effect = Exception("Failed to generate summary")

            with patch("streamlit.error") as mock_st_error:
                result, elapsed_time = service.generate_summary("question", DataFrame())

                # Should return None and display error
                assert result is None
                assert elapsed_time == 0.0
                mock_st_error.assert_called_once()
                assert "Error generating summary" in mock_st_error.call_args[0][0]

    @patch("streamlit.error")
    def test_remove_from_training_exception(self, mock_st_error):
        """Test that exceptions in remove_from_training are handled properly."""
        with patch.object(VannaService, "_setup_vanna"):
            service = VannaService()
            service.vn = MagicMock()
            service.vn.remove_training_data.side_effect = Exception("Failed to remove training data")

            result = service.remove_from_training("entry_id")

            # Should return False and display error
            assert result is False
            mock_st_error.assert_called_once()
            assert "Error removing training data" in mock_st_error.call_args[0][0]

    @patch("streamlit.error")
    def test_get_training_data_exception(self, mock_st_error):
        """Test that exceptions in get_training_data are handled properly."""
        with patch.object(VannaService, "_setup_vanna"):
            service = VannaService()
            service.vn = MagicMock()
            service.vn.get_training_data.side_effect = Exception("Failed to get training data")

            result = service.get_training_data()

            # Should return empty DataFrame and display error
            assert isinstance(result, DataFrame)
            assert result.empty
            mock_st_error.assert_called_once()
            assert "Error getting training data" in mock_st_error.call_args[0][0]

    @patch("streamlit.error")
    def test_train_exception(self, mock_st_error):
        """Test that exceptions in train are handled properly."""
        with patch.object(VannaService, "_setup_vanna"):
            service = VannaService()
            service.vn = MagicMock()
            # Make the actually called method raise the exception
            service.vn.add_question_sql.side_effect = Exception("Failed to train via add_question_sql") 

            result = service.train(question="q", sql="sql")

            # Should return False and display error
            assert result is False
            mock_st_error.assert_called_once()
            assert "Error training Vanna" in mock_st_error.call_args[0][0]


@pytest.mark.usefixtures("mock_streamlit_secrets")
class TestUtilityFunctionExceptions:
    def test_read_forbidden_from_json_exception(self):
        """Test that exceptions in read_forbidden_from_json are handled properly."""
        # Create a non-existent path to force an exception
        with patch("pathlib.Path.__truediv__", return_value=Path("/nonexistent/path")):
            # Should return empty lists
            tables, columns, tables_str = read_forbidden_from_json()

            assert tables == []
            assert columns == []
            assert tables_str == []

    @patch("streamlit.error")
    def test_write_to_file_and_training_exception(self, mock_st_error):
        """Test that exceptions in write_to_file_and_training are handled properly."""
        # Mock all file operations to prevent them from affecting real files
        m = mock_open(read_data='{"sample_queries": [], "sample_documents": []}')

        with (
            patch("utils.vanna_calls.VannaService.get_instance") as mock_get_instance,
            patch("builtins.open", m),
            patch("pathlib.Path.open", m),
            patch("json.load", return_value={"sample_queries": [], "sample_documents": []}),
        ):
            # Setup mock to raise exception
            mock_service = MagicMock()
            mock_get_instance.return_value = mock_service
            mock_service.train.side_effect = Exception("Training error")

            # Should catch the exception
            write_to_file_and_training({"question": "q", "query": "sql"})

            mock_st_error.assert_called_once()
            assert "Error writing to training_data.json" in mock_st_error.call_args[0][0]

    @patch("streamlit.error")
    def test_remove_from_file_training_exception(self, mock_st_error):
        """Test that exceptions in remove_from_file_training are handled properly."""
        # Mock all file operations to prevent them from affecting real files
        m = mock_open(read_data='{"sample_queries": [{"question": "q", "query": "sql"}], "sample_documents": []}')

        with (
            patch("utils.vanna_calls.VannaService.get_instance") as mock_get_instance,
            patch("builtins.open", m),
            patch("pathlib.Path.open", m),
            patch(
                "json.load",
                return_value={"sample_queries": [{"question": "q", "query": "sql"}], "sample_documents": []},
            ),
        ):
            # Setup mock to raise exception
            mock_service = MagicMock()
            mock_get_instance.return_value = mock_service
            mock_service.get_training_data.side_effect = Exception("Error getting training data")

            # Should catch the exception
            remove_from_file_training({"question": "q"})

            mock_st_error.assert_called_once()
            assert "Error removing entry from training_data.json" in mock_st_error.call_args[0][0]

    @patch("streamlit.error")
    def test_check_references_exception(self, mock_st_error):
        """Test that exceptions in check_references are handled properly."""
        with patch("utils.vanna_calls.read_forbidden_from_json", side_effect=Exception("Error reading forbidden")):
            # Should catch the exception
            check_references("SELECT * FROM table")

            mock_st_error.assert_called_once()
            assert "Error checking references" in mock_st_error.call_args[0][0]


@pytest.mark.usefixtures("mock_streamlit_secrets")
class TestTrainingFunctionExceptions:
    def test_training_plan_exception(self):
        """Test that exceptions in training_plan are handled properly."""
        with (
            patch("utils.vanna_calls.VannaService.get_instance") as mock_get_instance,
            patch("utils.vanna_calls.read_forbidden_from_json") as mock_read_forbidden,
            patch("streamlit.error") as mock_st_error,
        ):
            # Setup mocks
            mock_service = MagicMock()
            mock_get_instance.return_value = mock_service
            mock_read_forbidden.return_value = (["forbidden_table"], ["forbidden_column"], "'forbidden_table'")

            # Make run_sql raise an exception
            mock_service.run_sql.side_effect = Exception("SQL execution error")

            # Call the function and check exception is propagated
            with pytest.raises(Exception):
                training_plan()

    def test_training_plan_exception_get_plan(self):
        """Test that exceptions in get_training_plan_generic are handled properly."""
        with (
            patch("utils.vanna_calls.VannaService.get_instance") as mock_get_instance,
            patch("utils.vanna_calls.read_forbidden_from_json") as mock_read_forbidden,
            patch("streamlit.error") as mock_st_error,
        ):
            # Setup mocks
            mock_service = MagicMock()
            mock_get_instance.return_value = mock_service
            mock_read_forbidden.return_value = (["forbidden_table"], ["forbidden_column"], "'forbidden_table'")

            # Make run_sql return a DataFrame but get_training_plan_generic raise an exception
            mock_service.run_sql.return_value = pd.DataFrame({"column": ["value"]})
            mock_service.get_training_plan_generic.side_effect = Exception("Plan generation error")

            # Call the function and check exception is propagated
            with pytest.raises(Exception):
                training_plan()

    def test_training_plan_exception_train(self):
        """Test that exceptions in train with plan are handled properly."""
        with (
            patch("utils.vanna_calls.VannaService.get_instance") as mock_get_instance,
            patch("utils.vanna_calls.read_forbidden_from_json") as mock_read_forbidden,
            patch("streamlit.error") as mock_st_error,
        ):
            # Setup mocks
            mock_service = MagicMock()
            mock_get_instance.return_value = mock_service
            mock_read_forbidden.return_value = (["forbidden_table"], ["forbidden_column"], "'forbidden_table'")

            # Make run_sql return a DataFrame, get_training_plan_generic return a plan, but train raise an exception
            mock_service.run_sql.return_value = pd.DataFrame({"column": ["value"]})
            mock_service.get_training_plan_generic.return_value = "training plan"
            mock_service.train.side_effect = Exception("Training error")

            # Call the function
            with pytest.raises(Exception):
                training_plan()

    @patch("streamlit.error")
    def test_train_ddl_exception_connect(self, mock_st_error):
        """Test that exceptions in psycopg2.connect are handled properly."""
        with (
            patch("utils.vanna_calls.VannaService.get_instance"),
            patch("utils.vanna_calls.read_forbidden_from_json") as mock_read_forbidden,
            patch("psycopg2.connect", side_effect=Exception("Connection error")),
        ):
            # Make read_forbidden_from_json return valid values
            mock_read_forbidden.return_value = (["forbidden_table"], ["forbidden_column"], "'forbidden_table'")

            # Call the function
            train_ddl()

            # Verify error was displayed
            mock_st_error.assert_called_once()
            assert "Error training DDL" in mock_st_error.call_args[0][0]
            assert "Connection error" in mock_st_error.call_args[0][0]

    @patch("streamlit.error")
    def test_train_ddl_exception_execute(self, mock_st_error):
        """Test that exceptions in cursor.execute are handled properly."""
        with (
            patch("utils.vanna_calls.VannaService.get_instance"),
            patch("utils.vanna_calls.read_forbidden_from_json") as mock_read_forbidden,
            patch("psycopg2.connect") as mock_connect,
        ):
            # Setup mocks
            mock_read_forbidden.return_value = (["forbidden_table"], ["forbidden_column"], "'forbidden_table'")

            # Mock the cursor
            mock_cursor = MagicMock()
            mock_cursor.execute.side_effect = Exception("SQL execution error")

            # Mock the connection
            mock_conn = MagicMock()
            mock_conn.cursor.return_value = mock_cursor
            mock_connect.return_value = mock_conn

            # Call the function
            train_ddl()

            # Verify error was displayed
            mock_st_error.assert_called_once()
            assert "Error training DDL" in mock_st_error.call_args[0][0]
            assert "SQL execution error" in mock_st_error.call_args[0][0]

    @patch("streamlit.error")
    def test_train_file_exception(self, mock_st_error):
        """Test that exceptions in train_file are handled properly."""
        m = mock_open()
        m.side_effect = FileNotFoundError("[Errno 2] No such file or directory: 'mock_file'")

        with (
            patch("utils.vanna_calls.VannaService.get_instance"),
            patch("builtins.open", m),
            patch("pathlib.Path.open", m),
        ):
            # Call the function which should raise when trying to open the file
            train_file()

            # Verify error was displayed
            mock_st_error.assert_called_once()
            assert "Error training from file" in mock_st_error.call_args[0][0]

    @patch("streamlit.error")
    def test_train_file_exception_train(self, mock_st_error):
        """Test that exceptions in train during train_file are handled properly."""
        with (
            patch("utils.vanna_calls.VannaService.get_instance") as mock_get_instance,
            patch(
                "builtins.open",
                mock_open(read_data='{"sample_queries": [{"question": "q", "query": "sql"}], "sample_documents": []}'),
            ),
            patch("json.load") as mock_json_load,
        ):
            # Setup mocks
            mock_service = MagicMock()
            mock_get_instance.return_value = mock_service
            mock_service.train.side_effect = Exception("Training error")

            # Mock the JSON loading
            mock_json_load.return_value = {
                "sample_queries": [{"question": "q", "query": "sql"}],
                "sample_documents": [],
            }

            # Call the function
            train_file()

            # Verify error was displayed
            mock_st_error.assert_called_once()
            assert "Error training from file" in mock_st_error.call_args[0][0]
            assert "Training error" in mock_st_error.call_args[0][0]
