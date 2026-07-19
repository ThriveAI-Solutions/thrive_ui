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
from pydantic_ai.tools import ToolDefinition
from sqlalchemy.exc import SQLAlchemyError

from agent.codes.service import UnknownCodeSetError, VocabNotLoadedError
from agent.consent.gate import consent_required
from agent.consent.sql_gate import aggregate_measure_labels, classify_sql
from agent.consent.suppression import suppress_count
from agent.dataframe_adapters import run_sql_result_to_df
from agent.db.sql_context import schema_context_for_sql
from agent.deps import AgentDeps, QueryMeta
from agent.tools.sql_macros import expand_code_macros


_ROW_CAP = 500
# Match the curated-query path (analytics_adapter._CURATED_QUERY_TIMEOUT_S = 240).
# 240s is wide tolerance for the live HEALTHeLINK Redshift cluster, which
# under contention takes 30-90s for queries that complete in <1s when idle.
_TIMEOUT_S = 240

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


def _suppress_measure_cells(columns, rows, sql, dialect, threshold):
    """Small-cell-suppress the aggregate MEASURE columns of a result.

    Only columns the classifier counts as approved measures (COUNT / SUM(CASE
    0/1)) are suppressed; GROUP BY label columns (gender, year, ...) are left
    intact. Non-integer cells pass through untouched. Complementary suppression
    is not applied on the freeform path — additive partitions can't be reliably
    identified in arbitrary SQL — but every individual small count is hidden.
    """
    measure_labels = aggregate_measure_labels(sql, dialect=dialect)
    if not measure_labels:
        return columns, rows
    measure_idx = {i for i, name in enumerate(columns) if name in measure_labels}
    if not measure_idx:
        return columns, rows
    new_rows = []
    for row in rows:
        new_row = list(row)
        for i in measure_idx:
            cell = new_row[i]
            if isinstance(cell, int) and not isinstance(cell, bool):
                new_row[i] = suppress_count(cell, threshold=threshold)
        new_rows.append(new_row)
    return columns, new_rows


def run_sql(ctx: RunContext[AgentDeps], input: RunSqlInput) -> RunSqlResult:
    """Execute a read-only SELECT / WITH against the analytics warehouse.

    Use this when the curated clinical tools cannot answer the question
    (e.g., joining across domains in a non-standard way, multi-dimensional
    or unsupported breakdowns). Prefer get_patient_clinical_data for any
    per-patient clinical question, and prefer search_patients_by_criteria
    with `breakdown` for single-dimension population breakdowns.

    Filter clinical code sets via the {{codes:<set_id>}} macro (set_id from
    search_codes), e.g. `WHERE code IN {{codes:dx:diabetes-mellitus}}` —
    expanded server-side to dotted + undotted match-forms. NEVER use LIKE
    '%condition%' on a code column; resolve a set via search_codes instead.

    Results are capped at 500 rows. If truncated, refine the query for
    a smaller result.
    """
    original_sql = input.sql
    try:
        expansion = expand_code_macros(input.sql, ctx.deps.sqlite_session)
    except (UnknownCodeSetError, VocabNotLoadedError) as exc:
        raise ModelRetry(
            f"{exc} — call search_codes first and use an exact set_id it returned inside {{{{codes:<set_id>}}}}."
        ) from exc

    _ast_guard(expansion.sql)

    # Consent gate (#244): under enforcement, classify freeform SQL against
    # patient-bearing views. non_patient runs untouched; a pure aggregate
    # (COUNT / SUM(CASE 0/1) over non-identifying group keys, every UNION arm
    # included) runs then has its measure cells small-cell-suppressed;
    # anything row-level — or unparseable — is refused (fail-closed) and the
    # model is steered to the consent-gated curated tools. Role-aware; default
    # off until consent data/policy readiness.
    role = getattr(ctx.deps, "user_role", None)
    bypass_roles = getattr(ctx.deps, "consent_bypass_roles", frozenset())
    dialect = getattr(ctx.deps.analytics_db, "dialect", "postgres")
    suppress_measures = False
    if bool(getattr(ctx.deps, "enforce_consent", False)) and consent_required(role, bypass_roles):
        mode = classify_sql(expansion.sql, dialect=dialect)
        if mode in ("patient_row", "unparseable"):
            raise ModelRetry(
                "Consent enforcement blocks row-level freeform SQL against patient data. "
                "Use get_patient_clinical_data for per-patient questions or "
                "search_patients_by_criteria for population breakdowns — both apply consent. "
                "Only de-identified aggregate queries (COUNT / SUM(CASE .. 0/1) over "
                "non-identifying groupings) are permitted here, and their counts are suppressed below the disclosure threshold."
            )
        suppress_measures = mode == "aggregate"

    adapter = ctx.deps.analytics_db
    if adapter is None:
        raise ModelRetry("Analytics database is not configured for this session.")

    try:
        columns, rows, truncated = adapter.run_arbitrary_sql(sql=expansion.sql, row_cap=_ROW_CAP, timeout_s=_TIMEOUT_S)
    except ValueError as exc:
        # Adapter-level read-only guard tripped; re-raise as ModelRetry
        # so the LLM gets a chance to fix instead of crashing the run.
        raise ModelRetry(str(exc)) from exc
    except SQLAlchemyError as exc:
        # Warehouse rejected the SQL (UndefinedColumn, UndefinedTable,
        # syntax error, type mismatch, etc). Surface the underlying DB
        # message as a ModelRetry so the LLM can read it and fix the
        # query on the next turn instead of crashing the whole stream.
        # str(exc) on SQLAlchemyError already includes the wrapped
        # psycopg2 message and the offending SQL fragment.
        raise ModelRetry(f"SQL execution failed: {exc}") from exc

    if suppress_measures:
        columns, rows = _suppress_measure_cells(
            columns, rows, expansion.sql, dialect, int(getattr(ctx.deps, "small_cell_threshold", 11))
        )

    reliability = None
    if truncated:
        reliability = (
            f"Results truncated at {_ROW_CAP} rows. Refine the query "
            "(add filters, smaller date range, or aggregate) for completeness."
        )

    macro_note = None
    if expansion.expansions:
        macro_note = ", ".join(
            f"{{{{codes:{sid}}}}} expanded to {n} code match-forms" for sid, n in expansion.expansions.items()
        )

    result = RunSqlResult(
        sql=original_sql,
        columns=columns,
        rows=rows,
        row_count=len(rows),
        truncated=truncated,
        reliability_note=(f"{macro_note}; {reliability}" if macro_note and reliability else macro_note or reliability),
    )

    ctx.deps.last_dataframe = run_sql_result_to_df(result)
    ctx.deps.last_sql = expansion.sql
    ctx.deps.last_query_meta = QueryMeta(
        tool_name="run_sql",
        row_count=result.row_count,
        elapsed_ms=0,  # adapter doesn't time; the streaming layer fills this
        truncated=truncated,
    )
    return result


# Captured once at module load. The prepare hook is invoked per model turn
# within an agent run; pydantic-ai re-uses the same ToolDefinition object,
# so an in-place append grew the description by ~5800 chars per turn and
# pushed smaller models (gemma4:31b) past the point where they reliably
# retain correct tool names. Always rebuild from this stable base instead.
_RUN_SQL_BASE_DESCRIPTION = (run_sql.__doc__ or "").strip()


async def _augment_run_sql_description(
    ctx: RunContext[AgentDeps],
    tool_def: ToolDefinition,
) -> ToolDefinition:
    """Inject the fully-qualified schema catalog + few-shot SQL into run_sql's
    LLM-visible description. Schema prefix follows the configured analytics_db
    so production gets `dw.` and SQLite tests get bare names. Idempotent across
    multiple invocations on the same ToolDefinition."""
    adapter = getattr(ctx.deps, "analytics_db", None)
    prefix = getattr(adapter, "schema_prefix", "") if adapter is not None else ""
    tool_def.description = _RUN_SQL_BASE_DESCRIPTION + "\n\n" + schema_context_for_sql(prefix)
    return tool_def
