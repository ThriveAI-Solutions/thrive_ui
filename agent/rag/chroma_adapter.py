"""ChromaDB-backed RagAdapter for the agent.

Reuses the existing utils/chromadb_vector configuration but with
agent-specific collection names so it doesn't collide with the
Vanna-trained collections.
"""

from __future__ import annotations
from typing import List, Optional

import chromadb

from agent.rag.adapter import KBHit, RagAdapter

_AGENT_COLLECTION = "thrive_agent_kb"


class ChromaRagAdapter(RagAdapter):
    def __init__(self, client: chromadb.api.ClientAPI):
        self._client = client
        self._coll = client.get_or_create_collection(_AGENT_COLLECTION)

    def search(
        self,
        query: str,
        kind: Optional[str] = None,
        limit: int = 5,
    ) -> List[KBHit]:
        where = {"kind": kind} if kind else None
        result = self._coll.query(query_texts=[query], n_results=limit, where=where)
        hits: List[KBHit] = []
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        scores = result.get("distances", [[]])[0]
        for text, meta, dist in zip(docs, metas, scores):
            hits.append(
                KBHit(
                    text=text,
                    kind=(meta or {}).get("kind", "docs"),
                    score=1.0 - float(dist),
                    metadata=meta,
                )
            )
        return hits

    def upsert(self, doc_id: str, text: str, metadata: dict) -> None:
        self._coll.upsert(
            ids=[doc_id],
            documents=[text],
            metadatas=[metadata],
        )
