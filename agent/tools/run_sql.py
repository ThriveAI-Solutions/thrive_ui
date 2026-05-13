"""run_sql escape-hatch tool.

Phase 3 design §3.4. No role gating, no table whitelist — defenses are:
  1. sqlparse AST guard: one statement, must start with SELECT or WITH,
     no DDL/DML keywords, no system-table references.
  2. AnalyticsDbAdapter.run_arbitrary_sql: read-only session, row cap,
     statement timeout.
  3. 500-row cap, no escape hatch in Phase 3.
"""

from __future__ import annotations
from typing import Any, List, Optional

import sqlparse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_ai import RunContext
from pydantic_ai.exceptions import ModelRetry

from agent.dataframe_adapters import run_sql_result_to_df
from agent.deps import AgentDeps, QueryMeta


_ROW_CAP = 500
_TIMEOUT_S = 30

_FORBIDDEN_KEYWORDS = {
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "CREATE",
    "ALTER",
    "TRUNCATE",
    "GRANT",
    "REVOKE",
    "REPLACE",
    "MERGE",
    "VACUUM",
    "ANALYZE",
    "COPY",
}

_FORBIDDEN_TABLE_PREFIXES = ("pg_", "information_schema.")


class RunSqlInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sql: str = Field(..., description="A single read-only SQL statement.")

    @field_validator("sql")
    @classmethod
    def _strip_and_validate(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("sql must be a non-empty SELECT or WITH statement")
        return v


class RunSqlResult(BaseModel):
    sql: str
    columns: List[str]
    rows: List[List[Any]]
    row_count: int
    truncated: bool = False
    reliability_note: Optional[str] = None


def _ast_guard(sql: str) -> None:
    """Reject anything that isn't a single read-only SELECT/WITH.

    Raises ModelRetry on rejection — the LLM gets a chance to rewrite.
    """
    statements = [s for s in sqlparse.parse(sql) if str(s).strip()]
    if len(statements) != 1:
        raise ModelRetry(
            f"run_sql requires a single statement; got {len(statements)}. Submit one SELECT or WITH at a time."
        )

    parsed = statements[0]
    first_token = parsed.token_first(skip_cm=True)
    if first_token is None:
        raise ModelRetry("run_sql received an empty statement.")
    first_kw = first_token.normalized.upper()
    if first_kw not in ("SELECT", "WITH"):
        raise ModelRetry(f"run_sql is read-only; statement starts with {first_kw}, expected SELECT or WITH.")

    # Token-level scan: kw must appear as its own keyword token, not
    # as a substring of an identifier (e.g. CREATED_AT).
    for token in parsed.flatten():
        if token.is_keyword and token.normalized.upper() in _FORBIDDEN_KEYWORDS:
            raise ModelRetry(f"run_sql is read-only; statement contains write keyword {token.normalized.upper()}.")

    # System-table reference check — string-search is acceptable since
    # the AST step already confirmed structure is benign.
    lowered = str(parsed).lower()
    for prefix in _FORBIDDEN_TABLE_PREFIXES:
        if prefix in lowered:
            raise ModelRetry(
                f"run_sql cannot read from system tables (matched {prefix!r}); use the curated dw.* views instead."
            )


def run_sql(ctx: RunContext[AgentDeps], input: RunSqlInput) -> RunSqlResult:
    """Execute a read-only SELECT / WITH against the analytics warehouse.

    Use this when the curated clinical tools cannot answer the question
    (e.g., joining across domains in a non-standard way, ad-hoc aggregations).
    Prefer get_patient_clinical_data for any per-patient clinical question.

    Results are capped at 500 rows. If truncated, refine the query for
    a smaller result.
    """
    _ast_guard(input.sql)

    adapter = ctx.deps.analytics_db
    if adapter is None:
        raise ModelRetry("Analytics database is not configured for this session.")

    try:
        columns, rows, truncated = adapter.run_arbitrary_sql(sql=input.sql, row_cap=_ROW_CAP, timeout_s=_TIMEOUT_S)
    except ValueError as exc:
        # Adapter-level read-only guard tripped; re-raise as ModelRetry
        # so the LLM gets a chance to fix instead of crashing the run.
        raise ModelRetry(str(exc)) from exc

    reliability = None
    if truncated:
        reliability = (
            f"Results truncated at {_ROW_CAP} rows. Refine the query "
            "(add filters, smaller date range, or aggregate) for completeness."
        )

    result = RunSqlResult(
        sql=input.sql,
        columns=columns,
        rows=rows,
        row_count=len(rows),
        truncated=truncated,
        reliability_note=reliability,
    )

    ctx.deps.last_dataframe = run_sql_result_to_df(result)
    ctx.deps.last_sql = input.sql
    ctx.deps.last_query_meta = QueryMeta(
        tool_name="run_sql",
        row_count=result.row_count,
        elapsed_ms=0,  # adapter doesn't time; the streaming layer fills this
        truncated=truncated,
    )
    return result
