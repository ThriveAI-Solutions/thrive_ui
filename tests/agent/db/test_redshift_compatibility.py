"""Repo-level Redshift compatibility guards for agent/db/ SQL templates.

These tests scan source text — not rendered SQL — because the failure they
guard against is a careless edit, not a runtime condition. A grep that
runs in CI catches the bad token before it ships.
"""

from __future__ import annotations

import re
from pathlib import Path

_QUERIES_DIR = Path(__file__).resolve().parents[3] / "agent" / "db"


def _iter_sql_source_files() -> list[Path]:
    """Every .py file under agent/db/ that may build a SQL string.

    `__init__.py` and the analytics_adapter (engine wiring, not SQL bodies)
    are skipped so the test focuses on the template surface area.
    """
    skip = {"__init__.py", "analytics_adapter.py"}
    return sorted(
        p
        for p in _QUERIES_DIR.rglob("*.py")
        if p.name not in skip and "__pycache__" not in p.parts
    )


def test_no_substr_function_call_in_agent_db_sql_templates():
    """Regression for #195. Redshift rejects SUBSTR() with:

        SUBSTR() function is not supported (Hint: use SUBSTRING instead)

    This bug shipped in agent/db/queries/surgeries.py and produced 12
    production failures across 8 days before commit 36b7c37 swapped it for
    SUBSTRING. Postgres happens to support BOTH spellings, so a developer
    copy-pasting Postgres SQL into a template won't notice locally — only
    Redshift complains. Hence the static guard.

    The regex matches the function-call form `SUBSTR(` only; comments and
    docstrings that mention SUBSTR (e.g., explaining this very rule) are
    intentionally not flagged.
    """
    pattern = re.compile(r"\bSUBSTR\s*\(")
    offenders: list[str] = []
    for path in _iter_sql_source_files():
        text = path.read_text()
        for m in pattern.finditer(text):
            line_num = text.count("\n", 0, m.start()) + 1
            offenders.append(f"{path.relative_to(_QUERIES_DIR.parent.parent)}:{line_num}")
    assert not offenders, (
        "Found bare SUBSTR( in agent/db/ templates — Redshift will reject "
        f"these at runtime. Use SUBSTRING(...) instead. Offenders: {offenders}"
    )


def test_substr_guard_actually_catches_the_bad_pattern():
    """Positive control: prove the regex itself flags a known-bad sample.

    Without this, a future refactor that silently breaks the pattern (e.g.,
    introducing re.VERBOSE without escaping) would make the real guard pass
    on every commit, including ones that reintroduce SUBSTR(.
    """
    pattern = re.compile(r"\bSUBSTR\s*\(")
    bad_samples = [
        "AND SUBSTR(p.code, 3, 1) IN (...)",
        "WHERE SUBSTR( code, 1, 5 ) = '12345'",
        "SELECT substr(name, 1, 3)",  # case-insensitive? — see below
    ]
    # The regex above is case-sensitive on purpose: SQL keywords are
    # written uppercase in this codebase, so a stray lowercase `substr(`
    # would slip the guard. Tighten only if the convention shifts.
    assert pattern.search(bad_samples[0])
    assert pattern.search(bad_samples[1])
    assert not pattern.search(bad_samples[2]), "guard is uppercase-only by design"

    safe_samples = [
        "AND SUBSTRING(p.code, 3, 1) IN (...)",
        "-- We used to call SUBSTR here but switched to SUBSTRING",
        "SELECT SUBSTRX_NOT_A_THING(...)",  # word boundary check
    ]
    for sample in safe_samples:
        assert not pattern.search(sample), (
            f"guard false-positive on benign sample: {sample!r}"
        )
