import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--no-auto-cleanup",
        action="store_true",
        default=False,
        help="Disable automatic cleanup of test artifacts after test session",
    )


# Add the root directory to the Python path to ensure imports work correctly
@pytest.fixture(scope="session", autouse=True)
def add_root_to_path():
    """Add the project root directory to Python path."""
    root_dir = Path(__file__).parent.parent
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))
    yield


# Temporary directory for all test artifacts
@pytest.fixture(scope="session")
def test_temp_dir():
    """Create a temporary directory for all test artifacts that gets cleaned up after the session."""
    temp_dir = Path(tempfile.mkdtemp(prefix="thrive_test_"))
    yield temp_dir
    # Cleanup after all tests
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)


# In-memory ChromaDB client for tests
@pytest.fixture
def in_memory_chromadb_client():
    """Provides an in-memory ChromaDB client that doesn't persist to disk."""
    import chromadb

    return chromadb.Client()


# Test-specific temporary ChromaDB path
@pytest.fixture
def test_chromadb_path(test_temp_dir):
    """Provides a test-specific ChromaDB path in the temporary directory."""
    chromadb_path = test_temp_dir / f"chromadb_test_{os.getpid()}"
    chromadb_path.mkdir(parents=True, exist_ok=True)
    yield str(chromadb_path)
    # Cleanup happens automatically when test_temp_dir is cleaned up


# Mock streamlit secrets with temporary paths
@pytest.fixture
def mock_streamlit_secrets_with_temp_paths(test_chromadb_path):
    """Mock streamlit secrets using temporary paths for ChromaDB."""
    from unittest.mock import patch

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
            "rag_model": {"chroma_path": test_chromadb_path},
            "postgres": {
                "host": "localhost",
                "port": 5432,
                "database": "test_db",
                "user": "test_user",
                "password": "test_pass",
            },
            "security": {"allow_llm_to_see_data": False},
        },
    ):
        yield


# Backup and restore config files
@pytest.fixture(scope="session", autouse=True)
def backup_restore_config_files(test_temp_dir):
    """Backup config files before tests and restore them after."""
    # Path to the config directory
    config_dir = Path(__file__).parent.parent / "utils" / "config"

    # Make sure the config directory exists
    config_dir.mkdir(parents=True, exist_ok=True)

    # Create backup directory in temp space instead of tests directory
    backup_dir = test_temp_dir / "config_backup"
    backup_dir.mkdir(parents=True, exist_ok=True)

    # Files to backup
    files_to_backup = ["training_data.json", "forbidden_references.json"]

    # Backup files
    for file_name in files_to_backup:
        file_path = config_dir / file_name
        backup_path = backup_dir / file_name

        # Backup only if the file exists
        if file_path.exists():
            shutil.copy2(file_path, backup_path)
            print(f"Backed up {file_path} to {backup_path}")

    # Run tests
    yield

    # Restore files
    for file_name in files_to_backup:
        file_path = config_dir / file_name
        backup_path = backup_dir / file_name

        # Restore only if the backup exists
        if backup_path.exists():
            shutil.copy2(backup_path, file_path)
            print(f"Restored {file_path} from {backup_path}")


# Environment variable override for tests
@pytest.fixture(autouse=True)
def test_environment():
    """Set up test environment variables to prevent artifacts in project root."""
    original_env = os.environ.copy()

    # Set environment variables that might affect where files are created
    os.environ["PYTEST_RUNNING"] = "1"

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture(scope="session", autouse=True)
def cleanup_after_tests(request):
    """Run cleanup after all tests complete."""
    yield  # This runs before the test session starts

    # Check if user disabled auto-cleanup
    if request.config.getoption("--no-auto-cleanup"):
        if hasattr(request.config, "get_terminal_writer"):
            terminal_writer = request.config.get_terminal_writer()
            terminal_writer.line(
                "\n🔍 Auto-cleanup disabled. Use 'python scripts/cleanup_test_artifacts.py' to clean manually.",
                yellow=True,
            )
        else:
            print("\n🔍 Auto-cleanup disabled. Use 'python scripts/cleanup_test_artifacts.py' to clean manually.")
        return

    # This runs after the test session ends
    import sys
    from pathlib import Path

    # Import and run our cleanup function directly
    try:
        # Add the scripts directory to path temporarily
        scripts_dir = Path(__file__).parent.parent / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        # Import and run the cleanup function
        from cleanup_test_artifacts import cleanup_test_artifacts

        cleaned_items = cleanup_test_artifacts()

        # Use pytest's terminal writer for proper output
        if hasattr(request.config, "get_terminal_writer"):
            terminal_writer = request.config.get_terminal_writer()

            if cleaned_items:
                terminal_writer.line(f"\n🧹 Post-test cleanup: Removed {len(cleaned_items)} artifact(s)", green=True)
                for item in cleaned_items[:3]:  # Show first 3 items
                    terminal_writer.line(f"  - {item}")
                if len(cleaned_items) > 3:
                    terminal_writer.line(f"  ... and {len(cleaned_items) - 3} more items")
            else:
                terminal_writer.line("\n✅ Post-test cleanup: No artifacts found to clean", green=True)
        else:
            # Fallback to print if terminal writer not available
            if cleaned_items:
                print(f"\n🧹 Post-test cleanup: Removed {len(cleaned_items)} artifact(s)")
                for item in cleaned_items[:3]:
                    print(f"  - {item}")
                if len(cleaned_items) > 3:
                    print(f"  ... and {len(cleaned_items) - 3} more items")
            else:
                print("\n✅ Post-test cleanup: No artifacts found to clean")

    except Exception as e:
        if hasattr(request.config, "get_terminal_writer"):
            terminal_writer = request.config.get_terminal_writer()
            terminal_writer.line(f"\n⚠️  Post-test cleanup failed: {e}", red=True)
        else:
            print(f"\n⚠️  Post-test cleanup failed: {e}")
    finally:
        # Remove scripts dir from path
        if str(scripts_dir) in sys.path:
            sys.path.remove(str(scripts_dir))


# OIDC integration test fixture --------------------------------------------
@pytest.fixture
def in_memory_orm_session(monkeypatch):
    """In-memory SQLite ORM session with UserRoles seeded.

    Patches orm.models.SessionLocal and orm.models.engine so production code
    that imports SessionLocal (e.g. orm.functions.create_user) transparently
    uses this in-memory DB. Yields the SessionLocal class — call it to get
    a session. Tests typically use:

        with in_memory_orm_session() as session:
            session.query(User).all()
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from orm.models import Base, RoleTypeEnum, UserRole

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    # Seed the four UserRole rows so user_role_id FKs resolve.
    with TestingSessionLocal() as session:
        for role_name, description, role_enum in [
            ("Admin", "Administrator with full access", RoleTypeEnum.ADMIN),
            ("Doctor", "Physician with access to individual patient data", RoleTypeEnum.DOCTOR),
            ("Nurse", "Nurse with access to relevant patient data", RoleTypeEnum.NURSE),
            ("Patient", "Patient access only", RoleTypeEnum.PATIENT),
        ]:
            session.add(UserRole(role_name=role_name, description=description, role=role_enum))
        session.commit()

    monkeypatch.setattr("orm.models.SessionLocal", TestingSessionLocal)
    monkeypatch.setattr("orm.models.engine", engine)
    # orm.functions.SessionLocal is imported at module load time, so patch
    # there too. If a future caller does `from orm.models import SessionLocal`
    # in another module, add a similar monkeypatch line for that module.
    monkeypatch.setattr("orm.functions.SessionLocal", TestingSessionLocal)

    yield TestingSessionLocal

    engine.dispose()
