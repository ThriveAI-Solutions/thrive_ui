from unittest.mock import MagicMock
from agent.deps import AgentDeps
from agent.rag.adapter import RagAdapter, KBHit
from agent.tools.search_knowledge_base import search_knowledge_base, KBKind


class StubRag(RagAdapter):
    def __init__(self):
        self._docs = [
            KBHit(text="federated_results_v: lab results, LOINC ~50%", kind="schema", score=0.9),
            KBHit(text="example: SELECT ... WHERE source_id = ?", kind="examples", score=0.8),
        ]

    def search(self, query: str, kind: str | None = None, limit: int = 5) -> list[KBHit]:
        return [h for h in self._docs if kind is None or h.kind == kind][:limit]


def test_search_knowledge_base_unfiltered(monkeypatch):
    deps = MagicMock(spec=AgentDeps)
    deps.rag = StubRag()
    ctx = MagicMock()
    ctx.deps = deps
    hits = search_knowledge_base(ctx, query="lab results", kind=None)
    assert len(hits) == 2


def test_search_knowledge_base_filtered_to_schema():
    deps = MagicMock(spec=AgentDeps)
    deps.rag = StubRag()
    ctx = MagicMock()
    ctx.deps = deps
    hits = search_knowledge_base(ctx, query="anything", kind=KBKind.SCHEMA)
    assert len(hits) == 1
    assert hits[0].kind == "schema"
