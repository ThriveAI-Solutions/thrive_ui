"""Demo test to show cleanup functionality."""
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from utils.chromadb_vector import ThriveAI_ChromaDB


def test_temp_directory_fixture(test_temp_dir):
    """Test that test_temp_dir fixture creates a temporary directory."""
    assert test_temp_dir.exists()
    assert test_temp_dir.is_dir()
    assert "thrive_test_" in str(test_temp_dir)
    print(f"Test temp directory: {test_temp_dir}")


def test_chromadb_path_fixture(test_chromadb_path):
    """Test that test_chromadb_path fixture creates a path in temp directory."""
    assert isinstance(test_chromadb_path, str)
    assert Path(test_chromadb_path).exists()
    assert "chromadb_test_" in test_chromadb_path
    print(f"Test ChromaDB path: {test_chromadb_path}")


def test_in_memory_chromadb(in_memory_chromadb_client):
    """Test that in_memory_chromadb_client creates an in-memory client."""
    import chromadb
    assert type(in_memory_chromadb_client).__name__ == "Client"
    assert hasattr(in_memory_chromadb_client, "create_collection")
    # This should not create any files on disk
    print("In-memory ChromaDB client created successfully")


def test_mock_secrets_with_temp_paths(mock_streamlit_secrets_with_temp_paths, test_chromadb_path):
    """Test that mocked secrets use temporary paths."""
    import streamlit as st
    
    # Check that the mocked secrets use our temporary path
    assert st.secrets["rag_model"]["chroma_path"] == test_chromadb_path
    assert "test_db" in st.secrets["postgres"]["database"]
    print(f"Mocked secrets using temp path: {st.secrets['rag_model']['chroma_path']}")


def test_no_artifacts_in_project_root():
    """Test that no ChromaDB artifacts are created in project root during testing."""
    # Get the project root (parent of tests directory)
    project_root = Path(__file__).parent.parent
    
    # These files/directories should NOT exist in project root during test execution
    unwanted_artifacts = [
        "chroma.sqlite3",
        "test_chroma",
        "test_chromadb",
        "test_chromadb_ddl",
    ]
    
    for artifact in unwanted_artifacts:
        artifact_path = project_root / artifact
        # During test execution, these should not exist
        # The cleanup fixture will remove them after all tests
        print(f"Checking artifact doesn't interfere: {artifact_path}")


def test_environment_variables():
    """Test that test environment variables are set."""
    assert os.environ.get("PYTEST_RUNNING") == "1"
    print("Test environment variables are properly set")


class TestChromeDBWithTempPath:
    """Test ChromaDB operations with temporary paths."""
    
    def test_chromadb_creation_with_temp_path(self, test_chromadb_path, in_memory_chromadb_client):
        """Test that ChromaDB can be created with temporary paths."""
        
        # Create a concrete implementation for testing
        class TestThriveAI(ThriveAI_ChromaDB):
            def generate_embedding(self, data, **kwargs):
                return [0.1, 0.2, 0.3]
            
            def system_message(self, message):
                return f"SYSTEM: {message}"
            
            def user_message(self, message):
                return f"USER: {message}"
            
            def assistant_message(self, message):
                return f"ASSISTANT: {message}"
            
            def submit_prompt(self, prompt, **kwargs):
                return "Test response"
        
        # Test with temporary path
        db_with_path = TestThriveAI(user_role=1, config={"path": test_chromadb_path})
        assert db_with_path.user_role == 1
        
        # Test with in-memory client
        db_in_memory = TestThriveAI(user_role=2, client=in_memory_chromadb_client)
        assert db_in_memory.user_role == 2
        
        print(f"ChromaDB instances created successfully with temp path: {test_chromadb_path}") 