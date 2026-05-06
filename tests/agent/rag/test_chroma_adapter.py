import pytest
import chromadb
from agent.rag.chroma_adapter import ChromaRagAdapter


def test_chroma_adapter_round_trip(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    adapter = ChromaRagAdapter(client)
    adapter.upsert(
        doc_id="test-1",
        text="federated_results_v has lab data with LOINC codes",
        metadata={"kind": "schema"},
    )
    hits = adapter.search(query="lab results LOINC", kind="schema", limit=3)
    assert len(hits) == 1
    assert "LOINC" in hits[0].text
