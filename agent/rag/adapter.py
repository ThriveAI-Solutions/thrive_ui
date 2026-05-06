"""RAG adapter — abstracts ChromaDB and Milvus behind one interface.

Phase 1 only needs read access. Training/admin lives elsewhere and is
unchanged in this migration.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class KBHit:
    text: str
    kind: str  # "schema" | "examples" | "docs"
    score: float
    metadata: dict | None = None


class RagAdapter(ABC):
    @abstractmethod
    def search(
        self,
        query: str,
        kind: Optional[str] = None,
        limit: int = 5,
    ) -> List[KBHit]: ...

    @classmethod
    def from_streamlit_secrets(cls) -> "RagAdapter":
        """Build the configured adapter (Chroma or Milvus). Phase 1
        ships the Chroma path; Milvus support follows existing patterns
        from utils/milvus_vector.py."""
        # The actual ChromaRagAdapter / MilvusRagAdapter implementations
        # are wired in Phase 1 Task 20 (RAG seeding). Phase 1 tests use
        # the StubRag in tests/agent/tools/test_search_knowledge_base.py.
        raise NotImplementedError("Wired in Task 20.")
