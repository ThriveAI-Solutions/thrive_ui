"""search_codes — vocabulary lookup backed by the vocab_* tables.

Ported from chiron core/agent/tools/search_codes.py (spec 2026-07-03) onto
agent/codes/service.py (Task 1's DB-backed vocab search + code-set
expansion). Replaces the embedded-JSON loader that used to ship under
agent/codes/data/*.json — data now lives in the vocab_* tables, populated
via `scripts/import_vocab_dump.py`.

SNOMED coverage in this build is currently allergy-focused (Epic #203);
the curated category buckets live in agent/codes/allergies.py and are
reachable via curated set/code synonyms in the vocab DB, e.g. "penicillin
allergy", "peanut allergy", and "any food allergy".
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import RunContext

from agent.codes.service import VocabNotLoadedError, search_vocab
from agent.deps import AgentDeps


class CodeSearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vocabulary: Literal["icd10", "icd9", "loinc", "cvx", "rxnorm", "cpt", "snomed"]
    query: str
    limit: int = Field(default=20, le=50)


class CodeSetMatch(BaseModel):
    set_id: str
    name: str
    member_count: int
    sample_codes: list[str]


class CodeMatch(BaseModel):
    code: str
    display_name: str
    vocabulary: str
    is_active: bool


class CodeSearchResult(BaseModel):
    sets: list[CodeSetMatch]
    codes: list[CodeMatch]
    note: Optional[str] = None


def search_codes(
    ctx: RunContext[AgentDeps],
    query: CodeSearchInput,
) -> CodeSearchResult:
    """Look up clinical vocabulary codes and CODE SETS by name or abbreviation.

    When a result contains a set (`sets` is non-empty), prefer it over the
    individual `codes`: pass its `set_id` via `condition_sets` in
    search_patients_by_criteria, or the `{{codes:<set_id>}}` macro in
    run_sql, instead of copying individual codes by hand — the full member
    list is applied server-side and stays correct as the vocabulary is
    updated. Fall back to individual codes only when no set covers the
    concept, or for a per-patient domain filter that needs a small,
    specific code list.

    If the requested vocabulary has no data loaded yet, `codes` and `sets`
    come back empty and `note` explains why — this is a soft failure, not
    an error to retry: explain the gap to the user rather than looping on
    the same call.
    """
    try:
        result = search_vocab(
            ctx.deps.sqlite_session,
            vocabulary=query.vocabulary,
            query=query.query,
            limit=query.limit,
        )
    except VocabNotLoadedError as exc:
        return CodeSearchResult(sets=[], codes=[], note=str(exc))
    return CodeSearchResult(
        sets=[CodeSetMatch(**s.model_dump()) for s in result.sets],
        codes=[
            CodeMatch(
                code=c.code,
                display_name=c.display_name,
                vocabulary=c.vocabulary,
                is_active=c.is_active,
            )
            for c in result.codes
        ],
    )
