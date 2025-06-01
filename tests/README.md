# Thrive UI Tests

This directory contains tests for the Thrive UI codebase.

## Running Tests

To run all tests:

```bash
uv run python -m pytest
```

To run a specific test file:

```bash
uv run python -m pytest tests/utils/test_vanna_calls.py
```

To run a specific test class:

```bash
uv run python -m pytest tests/utils/test_vanna_calls.py::TestVannaService
```

To run a specific test:

```bash
uv run python -m pytest tests/utils/test_vanna_calls.py::TestVannaService::test_generate_questions
```

## Test Markers

You can run specific types of tests using markers:

```bash
# Run only unit tests
uv run python -m pytest -m unit

# Run only integration tests
uv run python -m pytest -m integration

# Run only slow tests
uv run python -m pytest -m slow
```

## Test Coverage

To get test coverage reports:

```bash
# Install pytest-cov first
uv pip install pytest-cov

# Run tests with coverage
uv run python -m pytest --cov=utils

# Generate an HTML coverage report
uv run python -m pytest --cov=utils --cov-report=html
```

Then open `htmlcov/index.html` in your browser to view the coverage report.

## Running Tests with Ollama and ChromaDB

The tests are configured to use mocks for Ollama and ChromaDB by default, so you don't need to have these services running to run the tests.

However, if you want to run integration tests that interact with actual Ollama and ChromaDB instances, make sure these services are running before executing the tests and use the following command:

```bash
uv run python -m pytest -m integration
```

## Debugging Tests

To debug tests, use the `-v` flag for verbose output:

```bash
uv run python -m pytest -v
```

For even more detailed output, add the `-s` flag to show print statements:

```bash
uv run python -m pytest -vs
```

# Test Setup and Cleanup Guide

This document explains the test infrastructure and cleanup mechanisms for the Thrive UI project.

## Problem Solved

Previously, tests were creating persistent artifacts in the project root:
- `chroma.sqlite3` files
- `chromadb/` directories with UUID subdirectories
- Various `test_chroma*` directories
- No cleanup after test runs

## Solution Overview

We've implemented a comprehensive test cleanup system using pytest fixtures that:
1. **Isolates test artifacts** in temporary directories
2. **Provides in-memory alternatives** for ChromaDB operations
3. **Automatically cleans up** after test sessions
4. **Prevents cross-test contamination**

## Automatic Cleanup

### Default Behavior
When you run `uv run pytest`, the system automatically:
- Runs all tests using temporary directories
- Cleans up any artifacts that slip through the isolation
- Shows a summary of what was cleaned up

Example output:
```
🧹 Post-test cleanup: Removed 3 artifact(s)
  - File: /path/to/chroma.sqlite3
  - ChromaDB UUID Directory: /path/to/uuid-directory
  - Temp Directory: /tmp/thrive_test_xyz
```

### Disabling Auto-Cleanup
If you need to inspect test artifacts for debugging:
```bash
uv run pytest --no-auto-cleanup
```

This will preserve all artifacts and show:
```
🔍 Auto-cleanup disabled. Use 'python scripts/cleanup_test_artifacts.py' to clean manually.
```

### Manual Cleanup
You can always run cleanup manually:
```bash
python scripts/cleanup_test_artifacts.py
```

## Key Fixtures

### `test_temp_dir` (Session-scoped)
Creates a temporary directory for all test artifacts that gets cleaned up after the session.

```python
def test_example(test_temp_dir):
    # test_temp_dir is a Path object pointing to a temporary directory
    test_file = test_temp_dir / "test.txt"
    test_file.write_text("test data")
    # Automatically cleaned up after session
```

### `test_chromadb_path` (Function-scoped)
Provides a test-specific ChromaDB path within the temporary directory.

```python
def test_chromadb(test_chromadb_path):
    # test_chromadb_path is a string path to a temporary ChromaDB directory
    # Use this instead of hardcoded paths
    config = {"chroma_path": test_chromadb_path}
```

### `in_memory_chromadb_client` (Function-scoped)
Provides an in-memory ChromaDB client that doesn't persist to disk.

```python
def test_chromadb_operations(in_memory_chromadb_client):
    # Use for tests that don't need persistence
    collection = in_memory_chromadb_client.create_collection("test")
```

### `mock_streamlit_secrets_with_temp_paths` (Function-scoped)
Mocks Streamlit secrets with temporary paths for ChromaDB.

```python
def test_with_secrets(mock_streamlit_secrets_with_temp_paths):
    # Streamlit secrets are automatically mocked with temp paths
    # Your code can use st.secrets normally
```

## Best Practices for Writing Tests

### 1. Use Temporary Paths
❌ **Don't do this:**
```python
def test_bad():
    config = {"chroma_path": "./test_chromadb"}  # Creates artifacts in project root
```

✅ **Do this:**
```python
def test_good(test_chromadb_path):
    config = {"chroma_path": test_chromadb_path}  # Uses temporary directory
```

### 2. Use In-Memory When Possible
❌ **Don't do this:**
```python
def test_bad():
    client = chromadb.PersistentClient(path="./test_db")  # Creates files
```

✅ **Do this:**
```python
def test_good(in_memory_chromadb_client):
    client = in_memory_chromadb_client  # No files created
```

### 3. Mock External Dependencies
✅ **Use the provided fixtures:**
```python
def test_with_mocked_secrets(mock_streamlit_secrets_with_temp_paths):
    # Streamlit secrets are automatically mocked
    service = VannaService.get_instance()  # Uses mocked secrets
```

### 4. Test Isolation
Each test should be independent and not rely on artifacts from other tests.

## Configuration Files

The system automatically backs up and restores configuration files:
- `utils/config/training_data.json`
- `utils/config/forbidden_references.json`

These are backed up before tests run and restored afterward, ensuring tests don't permanently modify your configuration.

## Debugging Test Artifacts

If you need to inspect what artifacts are being created:

1. **Run with no cleanup:**
   ```bash
   uv run pytest --no-auto-cleanup
   ```

2. **Check what would be cleaned:**
   ```bash
   python scripts/cleanup_test_artifacts.py
   ```

3. **Manually clean when done:**
   ```bash
   python scripts/cleanup_test_artifacts.py
   ```

## Troubleshooting

### Tests Creating Artifacts in Project Root
If you see artifacts in the project root after tests:
1. Check if your test is using hardcoded paths instead of fixtures
2. Ensure you're using `test_chromadb_path` or `in_memory_chromadb_client`
3. Run with `--no-auto-cleanup` to see what's being created

### Cleanup Not Working
1. Check if you have the `--no-auto-cleanup` flag set
2. Verify the cleanup script is executable: `chmod +x scripts/cleanup_test_artifacts.py`
3. Run manual cleanup: `python scripts/cleanup_test_artifacts.py`

### Test Failures Due to Artifacts
1. Run cleanup before tests: `python scripts/cleanup_test_artifacts.py`
2. Check if tests are interfering with each other (lack of isolation)
3. Use `pytest --lf` to run only last failed tests

## Summary

The new test infrastructure ensures:
- ✅ No artifacts left in project root
- ✅ Automatic cleanup after test runs
- ✅ Option to disable cleanup for debugging
- ✅ Proper test isolation
- ✅ Easy debugging and troubleshooting

Your tests should now run cleanly without leaving any trace in your project directory! 