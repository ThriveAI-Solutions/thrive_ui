import os
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