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