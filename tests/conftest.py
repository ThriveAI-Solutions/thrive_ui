import shutil
import sys
from pathlib import Path

import pytest


# Add the root directory to the Python path to ensure imports work correctly
@pytest.fixture(scope="session", autouse=True)
def add_root_to_path():
    """Add the project root directory to Python path."""
    root_dir = Path(__file__).parent.parent
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))
    yield


# Backup and restore config files
@pytest.fixture(scope="session", autouse=True)
def backup_restore_config_files():
    """Backup config files before tests and restore them after."""
    # Path to the config directory
    config_dir = Path(__file__).parent.parent / "utils" / "config"

    # Make sure the config directory exists
    config_dir.mkdir(parents=True, exist_ok=True)

    # Create backup directory
    backup_dir = Path(__file__).parent / "config_backup"
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
