"""Seed the agent's RAG collection with the curated v1 corpus.

Run:
    uv run python scripts/seed_agent_rag.py

Idempotent — uses upsert.
"""

from __future__ import annotations
import sys
import chromadb

from agent.rag.chroma_adapter import ChromaRagAdapter
from agent.rag.seed import all_seed_docs


def main(chroma_path: str = "./chromadb") -> None:
    client = chromadb.PersistentClient(path=chroma_path)
    adapter = ChromaRagAdapter(client)
    docs = all_seed_docs()
    for i, doc in enumerate(docs):
        doc_id = f"agent-seed-{i:03d}-{doc.get('view') or 'global'}"
        adapter.upsert(
            doc_id=doc_id,
            text=doc["text"],
            metadata={"kind": doc["kind"], "view": doc.get("view", "")},
        )
    print(f"Seeded {len(docs)} agent KB documents.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "./chromadb")
