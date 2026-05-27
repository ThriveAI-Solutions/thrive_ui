"""search_knowledge_base — RAG retrieval tool.

Per spec §7.6: kind-scoped retrieval over schema/examples/docs.
"""

from __future__ import annotations
from enum import Enum
from typing import List, Optional

from pydantic_ai import RunContext

from agent.deps import AgentDeps
from agent.rag.adapter import KBHit


class KBKind(str, Enum):
    SCHEMA = "schema"
    EXAMPLES = "examples"
    DOCS = "docs"


def search_knowledge_base(
    ctx: RunContext[AgentDeps],
    query: str,
    kind: Optional[KBKind] = None,
    limit: int = 5,
) -> List[KBHit]:
    """Retrieve relevant knowledge base entries.

    `kind` scopes the search:
    - schema: warehouse view documentation, column meanings
    - examples: prior question->SQL pairs
    - docs: general reference docs
    """
    return ctx.deps.rag.search(
        query=query,
        kind=kind.value if kind else None,
        limit=limit,
    )
