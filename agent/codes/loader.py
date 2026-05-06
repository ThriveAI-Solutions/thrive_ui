# agent/codes/loader.py
"""In-memory loader for the embedded vocabulary tables.

Per spec §7.4: search_codes is backed by JSON tables that ship with the
app. Loaded once per process; cached on the loader instance.

Search semantics: case-insensitive substring match on display_name AND on
the code itself. An empty query returns the head of the list (deterministic
sort by code).
"""

from __future__ import annotations
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List


_VOCABS = ("loinc", "cvx", "rxnorm", "icd10", "cpt")
_DATA_DIR = Path(__file__).parent / "data"


@dataclass(frozen=True)
class CodeMatch:
    code: str
    display_name: str
    vocabulary: str
    is_active: bool


class VocabLoader:
    def entries(self, vocabulary: str) -> list[dict]:
        if vocabulary not in _VOCABS:
            raise ValueError(f"unknown vocabulary: {vocabulary!r}")
        return _load(vocabulary)


@lru_cache(maxsize=None)
def _load(vocabulary: str) -> list[dict]:
    path = _DATA_DIR / f"{vocabulary}.json"
    return json.loads(path.read_text())


def search(*, vocabulary: str, query: str, limit: int = 20) -> List[CodeMatch]:
    if vocabulary not in _VOCABS:
        raise ValueError(f"unknown vocabulary: {vocabulary!r}")
    rows = _load(vocabulary)
    needle = (query or "").lower()
    if needle:
        matched = [r for r in rows if needle in r["display_name"].lower() or needle in r["code"].lower()]
    else:
        matched = list(rows)
    matched.sort(key=lambda r: (not r["display_name"].lower().startswith(needle), r["code"]))
    return [
        CodeMatch(
            code=r["code"],
            display_name=r["display_name"],
            vocabulary=vocabulary,
            is_active=r.get("is_active", True),
        )
        for r in matched[:limit]
    ]
