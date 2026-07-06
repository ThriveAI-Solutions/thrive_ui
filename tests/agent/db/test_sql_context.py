"""String-shape tests for agent.db.sql_context.schema_context_for_sql.

Verifies the rendered block: prefix qualifies every view name in BOTH
the catalog and the example SQL, header swaps based on prefix presence,
{p} placeholder is fully substituted, total output stays under budget.
"""

from __future__ import annotations


from agent.db.sql_context import schema_context_for_sql
from agent.rag.seed import SCHEMA_DOCS

# Bounds the SCHEMA BLOCK in isolation. The full LLM-visible run_sql
# description is base docstring + "\n\n" + this block; that assembled string
# is guarded separately by test_assembled_description_under_budget in
# tests/agent/tools/test_run_sql_description.py.
# Bumped 7000 → 8000 on 2026-07-06 for the EMPI sibling-expansion CTE in the
# per-patient example + identity doc (~7.7k current, ~300 headroom).
_BUDGET_CHARS = 8000


def test_dw_prefix_qualifies_every_view_in_catalog():
    out = schema_context_for_sql(schema_prefix="dw.")
    for doc in SCHEMA_DOCS:
        view = doc.get("view")
        if not view:
            continue
        assert f"dw.{view}" in out, f"expected dw.{view} in catalog block"


def test_dw_prefix_qualifies_views_in_example_sql():
    out = schema_context_for_sql(schema_prefix="dw.")
    # The first RUN_SQL_EXAMPLES entry references all four canonical views.
    for view in (
        "internal_patient_profile_v",
        "internal_source_reference_v",
        "federated_demographic_v",
        "metric_federated_data_v",
    ):
        assert f"dw.{view}" in out, f"expected dw.{view} in example SQL"


def test_empty_prefix_renders_bare_names():
    out = schema_context_for_sql(schema_prefix="")
    assert "dw." not in out, "no dw. should appear when schema_prefix is empty"
    # Bare view names still present.
    assert "internal_patient_profile_v" in out
    assert "federated_demographic_v" in out


def test_no_unrendered_placeholder_remains():
    for prefix in ("dw.", ""):
        out = schema_context_for_sql(schema_prefix=prefix)
        assert "{p}" not in out, f"unrendered placeholder for prefix={prefix!r}"


def test_header_demands_qualified_names_when_prefix_set():
    out = schema_context_for_sql(schema_prefix="dw.")
    lower = out.lower()
    assert "fully-qualified" in lower or "fully qualified" in lower
    assert "dw.<view>" in lower or "dw." in out


def test_header_when_prefix_empty_says_use_names_as_listed():
    out = schema_context_for_sql(schema_prefix="")
    lower = out.lower()
    assert "as listed" in lower or "exactly as listed" in lower


def test_output_under_budget():
    for prefix in ("dw.", ""):
        out = schema_context_for_sql(schema_prefix=prefix)
        assert len(out) <= _BUDGET_CHARS, (
            f"context grew past {_BUDGET_CHARS} chars (got {len(out)}); "
            "trim SCHEMA_DOCS / RUN_SQL_EXAMPLES or bump the budget."
        )


def test_per_patient_example_expands_source_ids_via_empi():
    """aea4b29 federated the curated tools across EMPI sibling source_ids,
    but the run_sql few-shot still modeled a bare `WHERE source_id =
    :source_id` filter on federated_results_v — under a header that says
    'copy these idioms; do not invent new join shapes'. The per-patient
    example must model the expansion (self-join internal_source_reference_v
    on patient_id, siblings at empi_rank != 99) and Redshift-safe NULLS LAST
    ordering (Postgres/Redshift sort NULLs first on DESC)."""
    for prefix in ("dw.", ""):
        out = schema_context_for_sql(schema_prefix=prefix)
        assert "empi_rank != 99" in out, f"missing EMPI sibling expansion (prefix={prefix!r})"
        assert "NULLS LAST" in out, f"missing NULLS LAST ordering (prefix={prefix!r})"


def test_no_example_filters_federated_view_by_bare_single_source_id():
    """The single-sid filter idiom silently drops the rest of a patient's
    chart (siblings hold most rows for multi-source patients). It must not
    appear anywhere in the block the LLM is told to copy."""
    for prefix in ("dw.", ""):
        out = schema_context_for_sql(schema_prefix=prefix)
        assert "WHERE source_id = :source_id" not in out


def test_identity_block_teaches_single_patient_expansion():
    """IDENTITY_DOCS covered only the counting direction (rank 1, never
    rank != 99 for people-counts). It must also teach the flip side: a
    single patient's chart spans sibling source_ids, so per-patient filters
    must expand through internal_source_reference_v."""
    out = schema_context_for_sql(schema_prefix="dw.")
    assert "sibling" in out.lower()


def test_question_kwarg_currently_ignored_but_accepted():
    # A2 will use this; today the body returns the same block regardless.
    a = schema_context_for_sql(schema_prefix="dw.", question="anything")
    b = schema_context_for_sql(schema_prefix="dw.", question=None)
    assert a == b
