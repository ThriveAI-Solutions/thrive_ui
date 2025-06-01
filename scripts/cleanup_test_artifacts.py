#!/usr/bin/env python3
"""
Cleanup script for test artifacts.

This script removes test artifacts that may have been left behind by tests.
It's safe to run at any time and will only remove known test artifacts.
"""

import shutil
import sys
import uuid
from pathlib import Path


def is_uuid_directory(path):
    """Check if a directory name is a valid UUID."""
    try:
        uuid.UUID(path.name)
        return True
    except ValueError:
        return False


def is_chromadb_directory(path):
    """Check if a directory contains ChromaDB artifacts."""
    if not path.is_dir():
        return False

    chromadb_files = ["data_level0.bin", "length.bin", "header.bin", "link_lists.bin"]
    return any((path / file).exists() for file in chromadb_files)


def cleanup_test_artifacts():
    """Remove test artifacts from the project directory."""
    # Get the project root directory
    project_root = Path(__file__).parent.parent

    # List of artifacts to clean up
    artifacts_to_clean = [
        "chroma.sqlite3",
        "test_chroma",
        "test_chromadb",
        "test_chromadb_ddl",
        ".pytest_cache",
    ]

    # Directories in temp that start with 'thrive_test_'
    temp_dirs = []
    import tempfile

    temp_root = Path(tempfile.gettempdir())
    if temp_root.exists():
        temp_dirs = list(temp_root.glob("thrive_test_*"))

    cleaned_items = []

    # Clean up known artifacts in project root
    for artifact in artifacts_to_clean:
        artifact_path = project_root / artifact
        if artifact_path.exists():
            if artifact_path.is_file():
                artifact_path.unlink()
                cleaned_items.append(f"File: {artifact_path}")
            elif artifact_path.is_dir():
                shutil.rmtree(artifact_path)
                cleaned_items.append(f"Directory: {artifact_path}")

    # Clean up UUID directories that contain ChromaDB artifacts
    for item in project_root.iterdir():
        if is_uuid_directory(item) and is_chromadb_directory(item):
            shutil.rmtree(item)
            cleaned_items.append(f"ChromaDB UUID Directory: {item}")

    # Clean up temporary test directories
    for temp_dir in temp_dirs:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            cleaned_items.append(f"Temp Directory: {temp_dir}")

    return cleaned_items


if __name__ == "__main__":
    print("Thrive UI Test Artifact Cleanup")
    print("=" * 40)

    try:
        cleaned_items = cleanup_test_artifacts()

        if cleaned_items:
            print("Cleaned up the following test artifacts:")
            for item in cleaned_items:
                print(f"  - {item}")
            print(f"\nCleanup complete. Removed {len(cleaned_items)} items.")
        else:
            print("No test artifacts found to clean up.")

        print("\nYour project directory is now clean of test artifacts.")

    except Exception as e:
        print(f"Error during cleanup: {e}")
        sys.exit(1)
