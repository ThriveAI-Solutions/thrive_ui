"""search_codes — vocabulary lookup for ICD-10 / LOINC / CVX / RxNorm / CPT.

Per spec §7.4: backed by embedded code reference tables (see
agent/codes/data/*.json).
"""

from __future__ import annotations
from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import RunContext

from agent.codes.loader import search as _search
from agent.deps import AgentDeps


class CodeSearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vocabulary: Literal["icd10", "loinc", "cvx", "rxnorm", "cpt"]
    query: str
    limit: int = Field(default=20, le=50)


class CodeMatch(BaseModel):
    code: str
    display_name: str
    vocabulary: str
    is_active: bool


def search_codes(
    ctx: RunContext[AgentDeps],
    query: CodeSearchInput,
) -> List[CodeMatch]:
    matches = _search(vocabulary=query.vocabulary, query=query.query, limit=query.limit)
    return [
        CodeMatch(
            code=m.code,
            display_name=m.display_name,
            vocabulary=m.vocabulary,
            is_active=m.is_active,
        )
        for m in matches
    ]
