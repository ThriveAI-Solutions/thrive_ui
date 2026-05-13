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


def test_prepare_injects_dw_qualified_views():
    ctx = _make_ctx("dw.")
    base = ToolDefinition(
        name="run_sql",
        description="Execute a read-only SELECT.",
        parameters_json_schema={"type": "object"},
    )
    out = _run(_augment_run_sql_description(ctx, base))
    assert out is not None
    assert "Execute a read-only SELECT." in out.description
    assert "dw.federated_demographic_v" in out.description
    assert "dw.internal_patient_profile_v" in out.description


def test_prepare_handles_empty_prefix_for_sqlite_tests():
    ctx = _make_ctx("")
    base = ToolDefinition(
        name="run_sql",
        description="Execute a read-only SELECT.",
        parameters_json_schema={"type": "object"},
    )
    out = _run(_augment_run_sql_description(ctx, base))
    assert out is not None
    assert "dw." not in out.description
    assert "internal_patient_profile_v" in out.description


def test_prepare_handles_missing_analytics_db_gracefully():
    """If analytics_db is None (misconfigured session), the hook should still
    return a definition rather than crash — run_sql will then fail at call
    time with the existing 'Analytics database is not configured' ModelRetry."""
    ctx = SimpleNamespace(deps=SimpleNamespace(analytics_db=None))
    base = ToolDefinition(
        name="run_sql",
        description="Execute a read-only SELECT.",
        parameters_json_schema={"type": "object"},
    )
    out = _run(_augment_run_sql_description(ctx, base))
    assert out is not None
    # No prefix means bare names — same as SQLite path.
    assert "internal_patient_profile_v" in out.description
