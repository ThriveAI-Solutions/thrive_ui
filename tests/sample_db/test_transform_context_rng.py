"""Regression coverage for deterministic per-key sample ETL RNG streams."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from scripts.sample_db.transformers.base import TransformContext

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SEQUENCE_SCRIPT = """
import json
from scripts.sample_db.transformers.base import TransformContext

rng = TransformContext(seed=42).rng("allergies")
print(json.dumps([rng.random() for _ in range(10)]))
"""


def _sequence_from_subprocess(hash_seed: str) -> list[float]:
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = hash_seed
    result = subprocess.run(
        [sys.executable, "-c", _SEQUENCE_SCRIPT],
        cwd=_REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def test_rng_sequence_is_stable_across_python_processes() -> None:
    assert _sequence_from_subprocess("1") == _sequence_from_subprocess("2")


def test_rng_streams_are_independent_per_key() -> None:
    context = TransformContext(seed=42)
    allergies = context.rng("allergies")
    medications = context.rng("medications")

    allergies_sequence = [allergies.random() for _ in range(20)]
    medications_sequence = [medications.random() for _ in range(5)]
    reference_rng = TransformContext(seed=42).rng("medications")

    assert allergies_sequence[:5] != medications_sequence
    assert medications_sequence == [reference_rng.random() for _ in range(5)]
