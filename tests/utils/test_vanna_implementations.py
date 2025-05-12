from unittest.mock import MagicMock, patch

import pytest
from psycopg2.extras import RealDictCursor

from utils.vanna_calls import (
    MyVannaAnthropic,
    MyVannaAnthropicChromaDB,
    MyVannaOllama,
    train_ddl,
)


# Mock the streamlit secrets for testing
@pytest.fixture
def mock_streamlit_secrets():
    with patch(
        "streamlit.secrets",
        new={
            "ai_keys": {
                "ollama_host": "http://localhost:11434",
                "ollama_model": "qwen2.5-coder:8b",
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


# Test for MyVannaAnthropic class
@pytest.mark.usefixtures("mock_streamlit_secrets")
class TestMyVannaAnthropic:
    def test_init(self):
        with (
            patch("utils.vanna_calls.VannaDB_VectorStore.__init__", return_value=None) as mock_vannadb_init,
            patch("utils.vanna_calls.Anthropic_Chat.__init__", return_value=None) as mock_anthropic_init,
        ):
            # Test initialization
            vanna_anthropic = MyVannaAnthropic()

            # Verify VannaDB_VectorStore was initialized with the right parameters
            mock_vannadb_init.assert_called_once()
            assert mock_vannadb_init.call_args[1]["vanna_model"] == "mock_vanna_model"
            assert mock_vannadb_init.call_args[1]["vanna_api_key"] == "mock_vanna_api"

            # Verify Anthropic_Chat was initialized with the right parameters
            mock_anthropic_init.assert_called_once()
            config = mock_anthropic_init.call_args[1]["config"]
            assert config["api_key"] == "mock_anthropic_api"
            assert config["model"] == "claude-3-sonnet-20240229"


# Test for MyVannaAnthropicChromaDB class
@pytest.mark.usefixtures("mock_streamlit_secrets")
class TestMyVannaAnthropicChromaDB:
    def test_init(self):
        with (
            patch("utils.vanna_calls.ChromaDB_VectorStore.__init__", return_value=None) as mock_chromadb_init,
            patch("utils.vanna_calls.Anthropic_Chat.__init__", return_value=None) as mock_anthropic_init,
        ):
            # Test initialization
            vanna_anthropic = MyVannaAnthropicChromaDB()

            # Verify ChromaDB_VectorStore was initialized with the right config
            mock_chromadb_init.assert_called_once()
            assert mock_chromadb_init.call_args[1]["config"]["path"] == "./chromadb"

            # Verify Anthropic_Chat was initialized with the right parameters
            mock_anthropic_init.assert_called_once()
            config = mock_anthropic_init.call_args[1]["config"]
            assert config["api_key"] == "mock_anthropic_api"
            assert config["model"] == "claude-3-sonnet-20240229"


# Test for MyVannaOllama class
@pytest.mark.usefixtures("mock_streamlit_secrets")
class TestMyVannaOllama:
    def test_init(self):
        with (
            patch("utils.vanna_calls.VannaDB_VectorStore.__init__", return_value=None) as mock_vannadb_init,
            patch("utils.vanna_calls.Ollama.__init__", return_value=None) as mock_ollama_init,
        ):
            # Test initialization
            vanna_ollama = MyVannaOllama()

            # Verify VannaDB_VectorStore was initialized with the right parameters
            mock_vannadb_init.assert_called_once()
            assert mock_vannadb_init.call_args[1]["vanna_model"] == "mock_vanna_model"
            assert mock_vannadb_init.call_args[1]["vanna_api_key"] == "mock_vanna_api"

            # Verify Ollama was initialized with the right config
            mock_ollama_init.assert_called_once()
            assert mock_ollama_init.call_args[1]["config"]["model"] == "qwen2.5-coder:8b"

    def test_log(self):
        with (
            patch("utils.vanna_calls.VannaDB_VectorStore.__init__", return_value=None),
            patch("utils.vanna_calls.Ollama.__init__", return_value=None),
            patch("utils.vanna_calls.logger.debug") as mock_logger,
        ):
            vanna_ollama = MyVannaOllama()
            vanna_ollama.log("Test message", "Test Title")

            # Verify logger.debug was called with correct arguments
            mock_logger.assert_called_once_with("%s: %s", "Test Title", "Test message")


# Test train_ddl function
@pytest.mark.usefixtures("mock_streamlit_secrets")
class TestTrainDDL:
    @patch("utils.vanna_calls.VannaService.get_instance")
    @patch("utils.vanna_calls.psycopg2.connect")
    @patch("utils.vanna_calls.read_forbidden_from_json")
    def test_train_ddl(self, mock_read_forbidden, mock_connect, mock_get_instance):
        # Mock the forbidden tables
        mock_read_forbidden.return_value = (["secret_table"], ["password"], "'secret_table'")

        # Mock the database connection and cursor
        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        # Mock the cursor fetchall method to return schema info
        mock_cursor.fetchall.return_value = [
            {
                "table_schema": "public",
                "table_name": "users",
                "column_name": "id",
                "data_type": "integer",
                "is_nullable": "NO",
            },
            {
                "table_schema": "public",
                "table_name": "users",
                "column_name": "name",
                "data_type": "varchar",
                "is_nullable": "YES",
            },
            {
                "table_schema": "public",
                "table_name": "posts",
                "column_name": "id",
                "data_type": "integer",
                "is_nullable": "NO",
            },
            {
                "table_schema": "public",
                "table_name": "posts",
                "column_name": "title",
                "data_type": "varchar",
                "is_nullable": "NO",
            },
        ]

        # Mock the VannaService instance
        mock_service = MagicMock()
        mock_get_instance.return_value = mock_service

        # Test the function
        train_ddl()

        # Verify psycopg2.connect was called with the correct arguments
        mock_connect.assert_called_once()
        call_args = mock_connect.call_args[1]
        assert call_args["host"] == "localhost"
        assert call_args["port"] == 5432
        assert call_args["database"] == "thrive"
        assert call_args["user"] == "postgres"
        assert call_args["password"] == "postgres"
        assert call_args["cursor_factory"] == RealDictCursor

        # Verify cursor.execute was called
        mock_cursor.execute.assert_called_once()

        # Verify VannaService.train was called twice (once for each table)
        assert mock_service.train.call_count == 2

        # Verify the calls contain the expected DDL statements
        first_call_args = mock_service.train.call_args_list[0][1]
        assert "CREATE TABLE users" in first_call_args["ddl"]
        assert "id integer NOT NULL" in first_call_args["ddl"]
        assert "name varchar NULL" in first_call_args["ddl"]

        second_call_args = mock_service.train.call_args_list[1][1]
        assert "CREATE TABLE posts" in second_call_args["ddl"]
        assert "id integer NOT NULL" in second_call_args["ddl"]
        assert "title varchar NOT NULL" in second_call_args["ddl"]

        # Verify cursor and connection were closed
        mock_cursor.close.assert_called_once()
        mock_conn.close.assert_called_once()
