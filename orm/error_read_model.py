"""Unified errors read model.

Merges three error sources into a single normalized :class:`ErrorRow` view
consumable by the Errors admin page (Feature Spec #94):

1. ``thrive_error_log`` rows (:class:`orm.models.ErrorLog`).
2. ``thrive_agent_run`` rows where ``success = False`` or ``error_type`` is
   non-null (agentic-flow failures).
3. JSONL fallback records produced by Feature Spec #89
   (:func:`utils.error_fallback_sink.read_fallback_records`).

This Phase 1 module defines the :class:`ErrorSource` enum and the
normalized :class:`ErrorRow` dataclass with a JSON-serializable
:meth:`ErrorRow.to_dict` method. Subsequent phases add the per-source
adapters and the public ``query_errors`` / ``count_errors_by_source``
entry points consumed by the UI Feature Spec.
"""

from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from sqlalchemy import func as sqla_func
from sqlalchemy import or_
from sqlalchemy.orm import Session

from orm.models import AgentRun, ErrorCategory, ErrorLog, ErrorSeverity, SessionLocal
from utils.error_fallback_sink import ErrorLoggingConfig, read_fallback_records
from utils.quick_logger import get_logger

logger = get_logger(__name__)


class ErrorSource(Enum):
    ERROR_LOG = "error_log"
    AGENT_RUN = "agent_run"
    FALLBACK_SINK = "fallback_sink"


# All three sources are queried by default unless query_errors caller passes a subset.
DEFAULT_SOURCES = frozenset({ErrorSource.ERROR_LOG, ErrorSource.AGENT_RUN, ErrorSource.FALLBACK_SINK})


@dataclass(frozen=True)
class ErrorRow:
    """One row in the unified errors view.

    Required fields: ``id``, ``source``, ``created_at``. The ``id`` is
    source-prefixed (e.g. ``"error_log:42"``, ``"agent_run:r-1"``,
    ``"fallback_sink:<created_at>:<error_type>:<hash>"``) so the UI can
    keep selection state across refreshes regardless of source.

    All other fields are optional because the three sources do not all
    provide every field — for instance, ``thrive_agent_run`` has no
    ``category`` / ``severity`` columns (synthesized by the adapter),
    and JSONL fallback records may lack ``message_id``.
    """

    id: str
    source: ErrorSource
    created_at: datetime
    user_id: int | None
    category: str | None
    severity: str | None
    error_type: str | None
    error_message: str | None
    stack_trace: str | None
    question: str | None
    generated_sql: str | None
    llm_provider: str | None
    llm_model: str | None
    context_data: str | None
    group_id: str | None
    run_id: str | None
    message_id: int | None
    auto_retry_attempted: bool | None
    retry_successful: bool | None
    retry_count: int | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict mirroring every dataclass field.

        Enum values become their string ``.value``; ``datetime`` becomes
        an ISO-8601 string. Optional fields preserve ``None``.
        """
        payload = dataclasses.asdict(self)
        payload["source"] = self.source.value
        payload["created_at"] = self.created_at.isoformat()
        return payload


# ── Source adapters ───────────────────────────────────────────────────────


def _row_from_error_log(r: ErrorLog) -> ErrorRow:
    """Map one ``thrive_error_log`` row to an :class:`ErrorRow`."""
    return ErrorRow(
        id=f"error_log:{r.id}",
        source=ErrorSource.ERROR_LOG,
        created_at=r.created_at,
        user_id=r.user_id,
        category=r.category,
        severity=r.severity,
        error_type=r.error_type,
        error_message=r.error_message,
        stack_trace=r.stack_trace,
        question=r.question,
        generated_sql=r.generated_sql,
        llm_provider=r.llm_provider,
        llm_model=r.llm_model,
        context_data=r.context_data,
        group_id=r.group_id,
        run_id=None,  # ErrorLog has no run_id column
        message_id=r.message_id,
        auto_retry_attempted=r.auto_retry_attempted,
        retry_successful=r.retry_successful,
        retry_count=r.retry_count,
    )


def _read_from_error_log(
    session: Session,
    since: datetime,
    until: datetime | None = None,
    categories: set[str] | None = None,
    severities: set[str] | None = None,
    user_id: int | None = None,
    search: str | None = None,
) -> list[ErrorRow]:
    """Read rows from ``thrive_error_log`` and map to :class:`ErrorRow`.

    Filters:
      - ``since``: ``created_at >= since`` (required).
      - ``until``: optional ``created_at <= until``.
      - ``categories``: optional ``category IN (...)``.
      - ``severities``: optional ``severity IN (...)``.
      - ``user_id``: optional ``user_id == user_id``.
      - ``search``: optional case-insensitive substring match against
        ``error_message`` OR ``question``.
    """
    query = session.query(ErrorLog).filter(ErrorLog.created_at >= since)
    if until is not None:
        query = query.filter(ErrorLog.created_at <= until)
    if categories:
        query = query.filter(ErrorLog.category.in_(categories))
    if severities:
        query = query.filter(ErrorLog.severity.in_(severities))
    if user_id is not None:
        query = query.filter(ErrorLog.user_id == user_id)
    if search:
        escaped = search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        pattern = f"%{escaped}%"
        query = query.filter(
            or_(
                ErrorLog.error_message.ilike(pattern, escape="\\"),
                ErrorLog.question.ilike(pattern, escape="\\"),
            )
        )
    return [_row_from_error_log(r) for r in query.all()]


_AGENT_RUN_CATEGORY = "agent_run"


def _agent_run_severity(status: str | None) -> str:
    """Map AgentRun.status to a synthetic severity for the unified view."""
    if status == "cap_reached":
        return "warning"
    return "error"


def _row_from_agent_run(r: AgentRun) -> ErrorRow:
    """Map one failed ``thrive_agent_run`` row to an :class:`ErrorRow`.

    ``category`` is the synthetic constant ``"agent_run"`` and
    ``severity`` is derived from ``status`` (cap_reached → warning,
    everything else → error). ``error_message`` comes from the
    ``AgentRun.error`` column.
    """
    return ErrorRow(
        id=f"agent_run:{r.run_id}",
        source=ErrorSource.AGENT_RUN,
        created_at=r.created_at,
        user_id=r.user_id,
        category=_AGENT_RUN_CATEGORY,
        severity=_agent_run_severity(r.status),
        error_type=r.error_type,
        error_message=r.error,
        stack_trace=r.stack_trace,
        question=r.question,
        generated_sql=None,  # AgentRun has no per-run generated_sql at the rollup
        llm_provider=r.llm_provider,
        llm_model=r.llm_model,
        context_data=None,
        group_id=r.group_id,
        run_id=r.run_id,
        message_id=r.final_message_id,
        auto_retry_attempted=None,
        retry_successful=None,
        retry_count=None,
    )


def _read_from_agent_run(
    session: Session,
    since: datetime,
    until: datetime | None = None,
    categories: set[str] | None = None,
    severities: set[str] | None = None,
    user_id: int | None = None,
    search: str | None = None,
) -> list[ErrorRow]:
    """Read failed ``thrive_agent_run`` rows and map to :class:`ErrorRow`.

    Failure filter is ``success == False`` OR ``error_type IS NOT NULL``.
    ``categories`` and ``severities`` filters are applied in Python after
    mapping because both fields are synthesized.
    """
    failure_filter = or_(
        AgentRun.success == False,  # noqa: E712
        AgentRun.error_type.isnot(None),
    )
    query = session.query(AgentRun).filter(
        AgentRun.created_at >= since,
        failure_filter,
    )
    if until is not None:
        query = query.filter(AgentRun.created_at <= until)
    if user_id is not None:
        query = query.filter(AgentRun.user_id == user_id)
    if search:
        escaped = search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        pattern = f"%{escaped}%"
        query = query.filter(
            or_(
                AgentRun.error.ilike(pattern, escape="\\"),
                AgentRun.question.ilike(pattern, escape="\\"),
            )
        )

    rows = [_row_from_agent_run(r) for r in query.all()]

    if categories:
        rows = [r for r in rows if r.category in categories]
    if severities:
        rows = [r for r in rows if r.severity in severities]
    return rows


def _row_from_fallback_record(record: dict) -> ErrorRow | None:
    """Map one fallback JSONL record dict to an :class:`ErrorRow`.

    Returns ``None`` for records that lack a parsable ``created_at`` —
    those are silently skipped (mirrors the behavior of
    :func:`utils.error_fallback_sink.read_fallback_records`).
    """
    created_raw = record.get("created_at")
    if not created_raw:
        return None
    try:
        when = datetime.fromisoformat(str(created_raw))
    except (TypeError, ValueError):
        return None

    error_type = record.get("error_type") or ""
    error_message = record.get("error_message") or ""
    synthetic_id = (
        f"fallback_sink:{created_raw}:{error_type}:"
        f"{hashlib.md5(error_message.encode('utf-8')).hexdigest()[:16]}"  # noqa: S324  # non-security: stable id for UI selection
    )

    return ErrorRow(
        id=synthetic_id,
        source=ErrorSource.FALLBACK_SINK,
        created_at=when,
        user_id=record.get("user_id"),
        category=record.get("category"),
        severity=record.get("severity"),
        error_type=record.get("error_type"),
        error_message=record.get("error_message"),
        stack_trace=record.get("stack_trace"),
        question=record.get("question"),
        generated_sql=record.get("generated_sql"),
        llm_provider=record.get("llm_provider"),
        llm_model=record.get("llm_model"),
        context_data=record.get("context_data"),
        group_id=record.get("group_id"),
        run_id=None,  # fallback records have no run_id at present
        message_id=record.get("message_id"),
        auto_retry_attempted=record.get("auto_retry_attempted"),
        retry_successful=record.get("retry_successful"),
        retry_count=record.get("retry_count"),
    )


def _read_from_fallback(
    since: datetime,
    until: datetime | None = None,
    categories: set[str] | None = None,
    severities: set[str] | None = None,
    user_id: int | None = None,
    search: str | None = None,
    *,
    fallback_config: ErrorLoggingConfig | None = None,
) -> list[ErrorRow]:
    """Read fallback JSONL records via :func:`read_fallback_records` and
    map them to :class:`ErrorRow`.

    Filters are applied in Python after read because the JSONL file is
    small relative to a query that has already paginated by time.
    """
    records = read_fallback_records(since, until, config=fallback_config)

    rows: list[ErrorRow] = []
    search_lower = search.lower() if search else None
    for record in records:
        row = _row_from_fallback_record(record)
        if row is None:
            continue
        if categories and row.category not in categories:
            continue
        if severities and row.severity not in severities:
            continue
        if user_id is not None and row.user_id != user_id:
            continue
        if search_lower is not None:
            em = (row.error_message or "").lower()
            q = (row.question or "").lower()
            if search_lower not in em and search_lower not in q:
                continue
        rows.append(row)
    return rows


# ── Public entry points ───────────────────────────────────────────────────


def query_errors(
    since: datetime,
    until: datetime | None = None,
    sources: set[ErrorSource] | None = None,
    categories: set[str] | None = None,
    severities: set[str] | None = None,
    user_id: int | None = None,
    search: str | None = None,
    limit: int | None = None,
    *,
    fallback_config: ErrorLoggingConfig | None = None,
) -> list[ErrorRow]:
    """Public entry point. Merge rows from every enabled source into one
    chronologically-descending list of :class:`ErrorRow`.

    ``sources`` defaults to :data:`DEFAULT_SOURCES` (all three). Per-source
    failures are isolated — a broken source contributes zero rows and
    logs a warning rather than blanking the other sources. ``limit`` is
    applied after the cross-source merge sort, not per source.
    """
    active_sources = sources if sources is not None else DEFAULT_SOURCES
    rows: list[ErrorRow] = []

    if ErrorSource.ERROR_LOG in active_sources or ErrorSource.AGENT_RUN in active_sources:
        try:
            with SessionLocal() as session:
                if ErrorSource.ERROR_LOG in active_sources:
                    try:
                        rows.extend(
                            _read_from_error_log(
                                session,
                                since,
                                until=until,
                                categories=categories,
                                severities=severities,
                                user_id=user_id,
                                search=search,
                            )
                        )
                    except Exception as exc:
                        logger.warning("ErrorLog adapter failed: %s", exc)
                if ErrorSource.AGENT_RUN in active_sources:
                    try:
                        rows.extend(
                            _read_from_agent_run(
                                session,
                                since,
                                until=until,
                                categories=categories,
                                severities=severities,
                                user_id=user_id,
                                search=search,
                            )
                        )
                    except Exception as exc:
                        logger.warning("AgentRun adapter failed: %s", exc)
        except Exception as exc:
            logger.warning("Failed to open session for DB error sources: %s", exc)

    if ErrorSource.FALLBACK_SINK in active_sources:
        try:
            rows.extend(
                _read_from_fallback(
                    since,
                    until=until,
                    categories=categories,
                    severities=severities,
                    user_id=user_id,
                    search=search,
                    fallback_config=fallback_config,
                )
            )
        except Exception as exc:
            logger.warning("Fallback adapter failed: %s", exc)

    rows.sort(key=lambda r: r.created_at, reverse=True)
    if limit is not None:
        rows = rows[:limit]
    return rows


def count_errors_by_source(
    since: datetime,
    until: datetime | None = None,
    *,
    fallback_config: ErrorLoggingConfig | None = None,
) -> dict[ErrorSource, int]:
    """Return per-source counts for the Errors UI's filter badges.

    Each source is counted under its own ``try/except``: a failure in one
    source contributes ``0`` for that source but does not affect the
    others. ``ErrorSource.AGENT_RUN`` counts only failure rows.
    """
    counts: dict[ErrorSource, int] = {s: 0 for s in ErrorSource}

    try:
        with SessionLocal() as session:
            try:
                q = session.query(sqla_func.count(ErrorLog.id)).filter(
                    ErrorLog.created_at >= since,
                )
                if until is not None:
                    q = q.filter(ErrorLog.created_at <= until)
                counts[ErrorSource.ERROR_LOG] = int(q.scalar() or 0)
            except Exception as exc:
                logger.warning("ErrorLog count failed: %s", exc)
            try:
                failure_filter = or_(
                    AgentRun.success == False,  # noqa: E712
                    AgentRun.error_type.isnot(None),
                )
                q = session.query(sqla_func.count(AgentRun.id)).filter(
                    AgentRun.created_at >= since,
                    failure_filter,
                )
                if until is not None:
                    q = q.filter(AgentRun.created_at <= until)
                counts[ErrorSource.AGENT_RUN] = int(q.scalar() or 0)
            except Exception as exc:
                logger.warning("AgentRun count failed: %s", exc)
    except Exception as exc:
        logger.warning("Failed to open session for count_errors_by_source: %s", exc)

    try:
        counts[ErrorSource.FALLBACK_SINK] = len(read_fallback_records(since, until, config=fallback_config))
    except Exception as exc:
        logger.warning("Fallback count failed: %s", exc)

    return counts


# ── Aggregates for the Errors-page KPI cards + charts ─────────────────────


_SQL_CATEGORIES = frozenset({ErrorCategory.SQL_GENERATION.value, ErrorCategory.SQL_EXECUTION.value})


@dataclass(frozen=True)
class ErrorAggregates:
    """Pre-rolled-up totals + chart series for the Errors page.

    Mirrors the four KPI cards from the legacy Admin Analytics → Error
    Analysis tab (``total`` / ``critical`` / ``sql_errors`` /
    ``retry_success_rate``) plus the three Plotly chart inputs
    (``over_time_by_category`` / ``by_category`` / ``by_severity``),
    rolled up across all three sources merged by :func:`query_errors`.
    """

    total: int
    critical: int
    sql_errors: int
    retry_attempted: int
    retry_successful: int
    retry_success_rate: float
    over_time_by_category: list[dict[str, Any]]
    by_category: dict[str, int]
    by_severity: list[dict[str, Any]]


def query_aggregates(
    since: datetime,
    until: datetime | None = None,
    *,
    fallback_config: ErrorLoggingConfig | None = None,
) -> ErrorAggregates:
    """Roll up errors across all three sources into KPI + chart summaries.

    Reads via :func:`query_errors` (no row limit) so the per-source
    ``try/except`` isolation already in that function handles partial
    source failures — a broken source contributes zero rows here rather
    than blanking the KPIs. The synthetic ``"agent_run"`` category
    appears as its own slice in :pyattr:`ErrorAggregates.by_category` and
    :pyattr:`ErrorAggregates.over_time_by_category`, matching the
    filter-chip experience already established by the Errors page.

    Cache this call at ~300s in the UI — chart data tolerates a longer
    TTL than the per-row drill-down list (60s).
    """
    rows = query_errors(
        since,
        until=until,
        fallback_config=fallback_config,
    )

    total = len(rows)
    critical = sum(1 for r in rows if r.severity == ErrorSeverity.CRITICAL.value)
    sql_errors = sum(1 for r in rows if r.category in _SQL_CATEGORIES)
    retry_attempted = sum(1 for r in rows if r.auto_retry_attempted)
    retry_successful = sum(1 for r in rows if r.retry_successful)
    retry_success_rate = round(retry_successful / retry_attempted * 100, 1) if retry_attempted else 0.0

    over_time_counts: dict[tuple[str, str], int] = {}
    by_category: dict[str, int] = {}
    by_severity_counts: dict[str, int] = {}

    for r in rows:
        date_key = r.created_at.strftime("%Y-%m-%d")
        cat_key = r.category or "unknown"
        sev_key = r.severity or "unknown"
        over_time_counts[(date_key, cat_key)] = over_time_counts.get((date_key, cat_key), 0) + 1
        by_category[cat_key] = by_category.get(cat_key, 0) + 1
        by_severity_counts[sev_key] = by_severity_counts.get(sev_key, 0) + 1

    over_time_by_category = sorted(
        ({"date": d, "category": c, "count": n} for (d, c), n in over_time_counts.items()),
        key=lambda x: (x["date"], x["category"]),
    )
    by_severity = sorted(
        ({"severity": s, "count": n} for s, n in by_severity_counts.items()),
        key=lambda x: x["severity"],
    )

    return ErrorAggregates(
        total=total,
        critical=critical,
        sql_errors=sql_errors,
        retry_attempted=retry_attempted,
        retry_successful=retry_successful,
        retry_success_rate=retry_success_rate,
        over_time_by_category=over_time_by_category,
        by_category=by_category,
        by_severity=by_severity,
    )
