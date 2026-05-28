from agent.rag.seed import EXAMPLES_DOCS, RUN_SQL_EXAMPLES, all_seed_docs


def test_examples_include_breakdown_routing():
    texts = " ".join(d["text"].lower() for d in EXAMPLES_DOCS)
    assert "breakdown" in texts and "by month" in texts


def test_run_sql_examples_excludes_breakdown_idiom_for_budget():
    # The breakdown SQL idiom is intentionally NOT a static run_sql example:
    # the tool's generated_sql handoff supplies a tailored template at runtime,
    # and the run_sql tool description must stay within its char budget
    # (see tests/agent/db/test_sql_context.py::test_output_under_budget).
    texts = " ".join(d["text"] for d in RUN_SQL_EXAMPLES)
    assert "DATE_TRUNC('month', dx.start_date)" not in texts


def test_all_seed_docs_includes_examples_and_run_sql():
    docs = all_seed_docs()
    for d in EXAMPLES_DOCS:
        assert d in docs
    for d in RUN_SQL_EXAMPLES:
        assert d in docs
