import chromadb
import pytest

from utils.chromadb_vector import ThriveAI_ChromaDB
from utils.milvus_vector import ThriveAI_Milvus


class ConcreteThriveAIChroma(ThriveAI_ChromaDB):
    """Concrete Chroma class with simple deterministic embedding for tests."""

    def __init__(self, user_role: int, client=None, config=None):
        super().__init__(user_role=user_role, client=client, config=config)

    def generate_embedding(self, data: str, **kwargs):
        # Simple character-count embedding (fixed length 8) to keep deterministic but weak semantics
        vec = [0.0] * 8
        for ch in data[:8]:
            vec[ord(ch) % 8] += 1.0
        return vec

    # Minimal LLM interface stubs not used in these tests
    def system_message(self, message: str):
        return message

    def user_message(self, message: str):
        return message

    def assistant_message(self, message: str):
        return message

    def submit_prompt(self, prompt, **kwargs):
        return ""


@pytest.fixture
def chroma_in_memory():
    return chromadb.Client()


@pytest.fixture
def milvus_lite(tmp_path, monkeypatch):
    """Give each test an isolated store that never calls configured embedding APIs."""
    store = ThriveAI_Milvus(
        user_role=1,
        config={
            "mode": "lite",
            "uri": str(tmp_path / "milvus.db"),
            "text_dim": 128,
            "collection_prefix": "test_milvus",
        },
    )

    def deterministic_embedding(text):
        return store._vectorizer.transform([text]).toarray().astype("float32")[0].tolist()

    monkeypatch.setattr(store, "generate_embedding", deterministic_embedding)
    yield store

    close = getattr(store._client, "close", None)
    if callable(close):
        close()


@pytest.mark.milvus
@pytest.mark.skipif(pytest.importorskip("pymilvus", reason="pymilvus not installed") is None, reason="pymilvus missing")
def test_milvus_hybrid_retrieval_prefers_keyword_match(milvus_lite):
    # Seed documentation
    relevant_doc = "Patient instructions contain UNIQUEKEY for discharge."
    other_doc = "General info without the special token."
    milvus_lite.add_documentation(relevant_doc)
    milvus_lite.add_documentation(other_doc)

    # Query using the keyword that BM25 should capture
    results = milvus_lite.get_related_documentation("Please find UNIQUEKEY in instructions")

    assert any("UNIQUEKEY" in doc for doc in results), "Milvus hybrid search should retrieve the keyword-matching doc"


@pytest.mark.milvus
@pytest.mark.skipif(pytest.importorskip("pymilvus", reason="pymilvus not installed") is None, reason="pymilvus missing")
def test_milvus_vs_chroma_top1_comparison(milvus_lite, chroma_in_memory):
    # Chroma in-memory store with weak dense embedding
    chroma = ConcreteThriveAIChroma(user_role=1, client=chroma_in_memory)

    # Seed a small corpus into both stores
    docs = [
        "Discharge steps include UNIQUE_A",
        "Follow-up schedule contains UNIQUE_B",
        "Medication notes mention UNIQUE_C",
        "General health advice",
        "Another general document",
    ]

    for d in docs:
        milvus_lite.add_documentation(d)
        chroma.add_documentation(d)

    queries = [
        ("Where is UNIQUE_A referenced?", "UNIQUE_A"),
        ("Find UNIQUE_B mention", "UNIQUE_B"),
        ("Locate details about UNIQUE_C", "UNIQUE_C"),
    ]

    milvus_top1_hits = 0
    chroma_top1_hits = 0

    for q, token in queries:
        m_res = milvus_lite.get_related_documentation(q)
        c_res = chroma.get_related_documentation(q)

        # Count top-1 accuracy: does the first result contain the token?
        if len(m_res) > 0 and token in m_res[0]:
            milvus_top1_hits += 1
        if len(c_res) > 0 and token in c_res[0]:
            chroma_top1_hits += 1

    assert milvus_top1_hits >= chroma_top1_hits, (
        f"Expected Milvus top-1 >= Chroma top-1, got Milvus={milvus_top1_hits}, Chroma={chroma_top1_hits}"
    )


@pytest.mark.milvus
@pytest.mark.skipif(pytest.importorskip("pymilvus", reason="pymilvus not installed") is None, reason="pymilvus missing")
def test_milvus_vs_chroma_mrr_at_k(milvus_lite, chroma_in_memory):
    chroma = ConcreteThriveAIChroma(user_role=1, client=chroma_in_memory)

    docs = [
        "Discharge steps include UNIQUE_A",
        "Follow-up schedule contains UNIQUE_B",
        "Medication notes mention UNIQUE_C",
        "General health advice",
        "Another general document",
    ]

    for d in docs:
        milvus_lite.add_documentation(d)
        chroma.add_documentation(d)

    queries = [
        ("Where is UNIQUE_A referenced?", "UNIQUE_A"),
        ("Find UNIQUE_B mention", "UNIQUE_B"),
        ("Locate details about UNIQUE_C", "UNIQUE_C"),
    ]

    def mrr_at_k(results_list, token, k=3):
        for idx, r in enumerate(results_list[:k], start=1):
            if token in r:
                return 1.0 / idx
        return 0.0

    milvus_mrr = 0.0
    chroma_mrr = 0.0

    for q, token in queries:
        m_res = milvus_lite.get_related_documentation(q)
        c_res = chroma.get_related_documentation(q)
        milvus_mrr += mrr_at_k(m_res, token, k=3)
        chroma_mrr += mrr_at_k(c_res, token, k=3)

    # Average
    milvus_mrr /= len(queries)
    chroma_mrr /= len(queries)

    assert milvus_mrr >= chroma_mrr, (
        f"Expected Milvus MRR>=Chroma MRR, got Milvus={milvus_mrr:.3f}, Chroma={chroma_mrr:.3f}"
    )
