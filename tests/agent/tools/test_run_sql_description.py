"""Verifies the run_sql prepare hook augments the tool description with the
fully-qualified schema catalog from agent.db.sql_context."""

from __future__ import annotations
import asyncio
import inspect
from types import SimpleNamespace

from pydantic_ai.tools import ToolDefinition

from agent.tools.run_sql import _augment_run_sql_description


def _make_ctx(schema_prefix: str):
    deps = SimpleNamespace(analytics_db=SimpleNamespace(schema_prefix=schema_prefix))
    # RunContext attribute access is duck-typed for our hook's needs.
    return SimpleNamespace(deps=deps)


def _run(coro_or_value):
    if inspect.isawaitable(coro_or_value):
        return asyncio.run(coro_or_value)
    return coro_or_value


def _base_tool_def() -> ToolDefinition:
    # The hook ignores tool_def.description and rebuilds from the captured
    # run_sql docstring; passing the real docstring here mirrors what
    # pydantic-ai actually does at registration time.
    from agent.tools.run_sql import _RUN_SQL_BASE_DESCRIPTION

    return ToolDefinition(
        name="run_sql",
        description=_RUN_SQL_BASE_DESCRIPTION,
        parameters_json_schema={"type": "object"},
    )


def test_prepare_injects_dw_qualified_views():
    ctx = _make_ctx("dw.")
    out = _run(_augment_run_sql_description(ctx, _base_tool_def()))
    assert out is not None
    assert "Execute a read-only SELECT / WITH against the analytics warehouse" in out.description
    assert "dw.federated_demographic_v" in out.description
    assert "dw.internal_patient_profile_v" in out.description


def test_prepare_handles_empty_prefix_for_sqlite_tests():
    ctx = _make_ctx("")
    out = _run(_augment_run_sql_description(ctx, _base_tool_def()))
    assert out is not None
    assert "dw." not in out.description
    assert "internal_patient_profile_v" in out.description


def test_prepare_handles_missing_analytics_db_gracefully():
    """If analytics_db is None (misconfigured session), the hook should still
    return a definition rather than crash — run_sql will then fail at call
    time with the existing 'Analytics database is not configured' ModelRetry."""
    ctx = SimpleNamespace(deps=SimpleNamespace(analytics_db=None))
    out = _run(_augment_run_sql_description(ctx, _base_tool_def()))
    assert out is not None
    # No prefix means bare names — same as SQLite path.
    assert "internal_patient_profile_v" in out.description


def test_assembled_description_under_budget():
    """Guard the FULL LLM-visible description (base docstring + schema block),
    not just the schema block in isolation. test_sql_context.py bounds only
    schema_context_for_sql; but the hook ships base + "\\n\\n" + schema, and an
    oversized description is what actually degrades small-model (gemma4:31b)
    tool-name retention. Bounding the assembled string is the invariant that
    matches reality. Budget has ~500 chars of headroom over current (~6.5k);
    if it trips, trim SCHEMA_DOCS / RUN_SQL_EXAMPLES or the run_sql docstring."""
    _ASSEMBLED_BUDGET_CHARS = 7500
    for prefix in ("dw.", ""):
        ctx = _make_ctx(prefix)
        out = _run(_augment_run_sql_description(ctx, _base_tool_def()))
        assert len(out.description) <= _ASSEMBLED_BUDGET_CHARS, (
            f"assembled run_sql description grew past {_ASSEMBLED_BUDGET_CHARS} "
            f"chars (got {len(out.description)} at prefix {prefix!r})"
        )


def test_prepare_is_idempotent_across_repeated_invocations():
    """Pydantic-ai re-uses the same ToolDefinition object across the model
    turns within one agent run. The hook MUST rebuild from a stable base
    rather than appending to whatever it was last called with — otherwise
    description grows unbounded across turns and smaller models lose the
    plot on tool names. Live regression caught by gemma4:31b on 2026-05-13."""
    ctx = _make_ctx("dw.")
    tool_def = _base_tool_def()
    first = _run(_augment_run_sql_description(ctx, tool_def))
    first_len = len(first.description)
    # Re-invoke on the same (already-mutated) tool_def object several times.
    for _ in range(5):
        out = _run(_augment_run_sql_description(ctx, tool_def))
        assert len(out.description) == first_len, (
            f"description grew across invocations ({first_len} → {len(out.description)}); "
            "prepare hook is not idempotent"
        )
        # Schema header must appear exactly once — never nested.
        assert out.description.count("SCHEMA REFERENCE") == 1
