# agent/codes/loader.py
"""In-memory loader for the embedded vocabulary tables.

Per spec §7.4: search_codes is backed by JSON tables that ship with the
app. Loaded once per process; cached on the loader instance.

Two search layers:

1. **Synonym map** (`synonyms.json`): a curated abbreviation / colloquial
   → canonical → code-family mapping. Doctors typing "HTN" or patients
   typing "high blood pressure" both hit the same hypertension code list,
   sidestepping the substring-match limitation that canonical display
   names ("Essential primary hypertension") don't contain abbreviations
   or lay terms. First match wins; codes are returned in the order the
   map lists them so the most clinically relevant code appears first.
2. **Substring fallback** (original behavior): case-insensitive substring
   match on display_name AND on the code itself. Handles technical terms
   not in the synonym map (e.g., "essential primary hypertension" matches
   the display name verbatim) and any vocabulary that doesn't have a
   synonym entry yet. An empty query returns the head of the list
   (deterministic sort by code).
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


@lru_cache(maxsize=1)
def _load_synonyms() -> dict:
    """Load the curated synonym → code-family map. Returns {} if the file
    is missing so the substring-search fallback still works in old
    check-outs. Schema documented in agent/codes/data/synonyms.json::_meta."""
    path = _DATA_DIR / "synonyms.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    # Drop the _meta block — it's documentation, not data.
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def _resolve_synonym(vocabulary: str, query: str) -> list[str]:
    """If `query` matches a canonical key OR any listed synonym for the
    given vocabulary, return the curated code list. Empty list otherwise."""
    by_vocab = _load_synonyms().get(vocabulary, {})
    if not by_vocab or not query:
        return []
    needle = query.lower().strip()
    # 1. Direct canonical hit.
    if needle in by_vocab:
        return list(by_vocab[needle].get("codes", []))
    # 2. Synonym hit — case-insensitive across the entry's synonyms list.
    for entry in by_vocab.values():
        synonyms = [s.lower() for s in entry.get("synonyms", [])]
        if needle in synonyms:
            return list(entry.get("codes", []))
    return []


def search(*, vocabulary: str, query: str, limit: int = 20) -> List[CodeMatch]:
    if vocabulary not in _VOCABS:
        raise ValueError(f"unknown vocabulary: {vocabulary!r}")
    rows = _load(vocabulary)
    needle = (query or "").lower().strip()

    # Synonym layer first. Skip codes that aren't actually present in the
    # vocabulary file (so a stale synonym map entry never returns a
    # phantom code). Order is preserved from the map so the most clinically
    # relevant code appears first.
    synonym_codes = _resolve_synonym(vocabulary, needle)
    if synonym_codes:
        by_code = {r["code"]: r for r in rows}
        matched_rows = [by_code[c] for c in synonym_codes if c in by_code]
        if matched_rows:
            return [
                CodeMatch(
                    code=r["code"],
                    display_name=r["display_name"],
                    vocabulary=vocabulary,
                    is_active=r.get("is_active", True),
                )
                for r in matched_rows[:limit]
            ]

    # Substring fallback for technical terms / unknown queries.
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
