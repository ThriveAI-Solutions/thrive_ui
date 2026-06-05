"""Durable fallback sink for error records.

When :func:`orm.logging_functions.log_error` cannot write to the SQLite-backed
``thrive_error_log`` table (file locked, disk full, connection unavailable),
the error record is appended here instead so it is never silently dropped.

The fallback file is a rotating JSONL log — one JSON object per line — that
mirrors :class:`orm.models.ErrorLog`'s columns. The unified errors read
model (Feature Spec #93) and the Errors admin page (Feature Spec #94)
consume this file alongside ``thrive_error_log`` rows.

Fallback record schema
----------------------
Each line in the fallback file is a JSON object. ``source`` is set by the
writer (Feature Spec #93's normalizer); the other fields mirror
``orm.models.ErrorLog``:

- ``created_at``           ISO-8601 timestamp string
- ``source``               constant ``"fallback_sink"`` (added by the read model)
- ``category``             see ``orm.models.ErrorCategory``
- ``severity``             see ``orm.models.ErrorSeverity``
- ``error_type``           exception class name
- ``error_message``        exception message
- ``stack_trace``          str | null
- ``question``             str | null
- ``generated_sql``        str | null
- ``llm_provider``         str | null
- ``llm_model``            str | null
- ``context_data``         JSON string | null
- ``user_id``              int | null
- ``message_id``           int | null
- ``group_id``             str | null  — message-flow correlation UUID
- ``auto_retry_attempted`` bool | null
- ``retry_successful``     bool | null
- ``retry_count``          int | null

Config (``secrets.toml``)
-------------------------
    [error_logging]
    fallback_path = "utils/logs/error_fallback.jsonl"
    fallback_max_bytes = 5000000
    fallback_backup_count = 5

Absent section ⇒ defaults ⇒ behavior unchanged.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from sqlalchemy import and_, or_

from orm.models import ErrorLog, SessionLocal

_DEFAULT_LOG_DIR = Path(__file__).with_name("logs")
_DEFAULT_PATH = _DEFAULT_LOG_DIR / "error_fallback.jsonl"
_DEFAULT_MAX_BYTES = 5_000_000
_DEFAULT_BACKUP_COUNT = 5

_INITIALIZED_LOGGER_NAMES: set[str] = set()


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _get_streamlit_section() -> dict | None:
    """Return the ``[error_logging]`` section from ``st.secrets``.

    Exposed as a module-level helper so tests can patch it. Raises if
    streamlit is unavailable; ``ErrorLoggingConfig.from_streamlit`` catches
    that and returns defaults.
    """
    import streamlit as st  # local: streamlit may not be importable in unit tests

    return dict(st.secrets.get("error_logging", {}))


@dataclass(frozen=True)
class ErrorLoggingConfig:
    fallback_path: Path = _DEFAULT_PATH
    fallback_max_bytes: int = _DEFAULT_MAX_BYTES
    fallback_backup_count: int = _DEFAULT_BACKUP_COUNT

    @classmethod
    def from_secrets(cls, section: dict | None) -> "ErrorLoggingConfig":
        section = dict(section or {})
        return cls(
            fallback_path=Path(section.get("fallback_path", _DEFAULT_PATH)),
            fallback_max_bytes=_coerce_int(
                section.get("fallback_max_bytes", _DEFAULT_MAX_BYTES),
                _DEFAULT_MAX_BYTES,
            ),
            fallback_backup_count=_coerce_int(
                section.get("fallback_backup_count", _DEFAULT_BACKUP_COUNT),
                _DEFAULT_BACKUP_COUNT,
            ),
        )

    @classmethod
    def from_streamlit(cls) -> "ErrorLoggingConfig":
        try:
            section = _get_streamlit_section()
        except Exception:
            section = None
        return cls.from_secrets(section)


def _get_logger(config: ErrorLoggingConfig) -> logging.Logger:
    """Lazy-init a dedicated logger writing to the configured rotating file."""
    name = f"thrive.error_fallback.{config.fallback_path}"
    logger = logging.getLogger(name)
    if name not in _INITIALIZED_LOGGER_NAMES:
        logger.propagate = False
        logger.setLevel(logging.INFO)
        config.fallback_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            str(config.fallback_path),
            maxBytes=config.fallback_max_bytes,
            backupCount=config.fallback_backup_count,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        _INITIALIZED_LOGGER_NAMES.add(name)
    return logger


def _reset_handler_cache() -> None:
    """Close + remove cached fallback handlers. Test helper; not for app use."""
    for name in list(_INITIALIZED_LOGGER_NAMES):
        logger = logging.getLogger(name)
        for handler in list(logger.handlers):
            try:
                handler.close()
            except Exception:
                pass
            logger.removeHandler(handler)
    _INITIALIZED_LOGGER_NAMES.clear()


def write_fallback_record(payload: dict, *, config: ErrorLoggingConfig | None = None) -> None:
    """Append one JSON record to the rotating fallback file.

    Never raises: any failure (serialization, I/O, missing config) is
    swallowed so the caller's error-handling path cannot itself be broken
    by a logging failure.
    """
    try:
        cfg = config or ErrorLoggingConfig.from_streamlit()
        line = json.dumps(payload, default=str)
        _get_logger(cfg).info(line)
    except Exception:
        pass


def _iter_fallback_files(config: ErrorLoggingConfig) -> Iterator[Path]:
    """Yield the current fallback file followed by each rotated backup that exists."""
    base = config.fallback_path
    if base.exists():
        yield base
    for i in range(1, config.fallback_backup_count + 1):
        rotated = base.with_name(f"{base.name}.{i}")
        if rotated.exists():
            yield rotated


def read_fallback_records(
    since: datetime,
    until: datetime | None = None,
    limit: int | None = None,
    *,
    config: ErrorLoggingConfig | None = None,
) -> list[dict]:
    """Return fallback records with ``created_at`` ∈ [since, until], sorted ascending.

    Reads the current JSONL file and every rotated backup (``.1``…``.N``).
    Records that are blank, malformed, not dicts, lack ``created_at``, or
    have an unparsable ``created_at`` are silently skipped. File-open
    failures on a single backup are tolerated so a corrupt file cannot
    abort the read. ``limit`` is applied after sorting.
    """
    cfg = config or ErrorLoggingConfig.from_streamlit()
    records: list[dict] = []

    for path in _iter_fallback_files(cfg):
        try:
            with path.open("r", encoding="utf-8") as fp:
                for line in fp:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except (json.JSONDecodeError, ValueError):
                        continue
                    if not isinstance(payload, dict):
                        continue
                    created_raw = payload.get("created_at")
                    if not created_raw:
                        continue
                    try:
                        when = datetime.fromisoformat(str(created_raw))
                    except (TypeError, ValueError):
                        continue
                    if when < since:
                        continue
                    if until is not None and when > until:
                        continue
                    records.append(payload)
        except OSError:
            continue

    records.sort(key=lambda r: str(r.get("created_at", "")))
    if limit is not None:
        records = records[:limit]
    return records


_REQUIRED_FOR_DB = ("created_at", "category", "severity", "error_type", "error_message")


def _build_error_log_row(payload: dict, when: datetime) -> ErrorLog:
    return ErrorLog(
        created_at=when,
        user_id=payload.get("user_id"),
        message_id=payload.get("message_id"),
        group_id=payload.get("group_id"),
        category=payload["category"],
        severity=payload["severity"],
        error_type=payload["error_type"],
        error_message=payload["error_message"],
        stack_trace=payload.get("stack_trace"),
        question=payload.get("question"),
        generated_sql=payload.get("generated_sql"),
        llm_provider=payload.get("llm_provider"),
        llm_model=payload.get("llm_model"),
        context_data=payload.get("context_data"),
        auto_retry_attempted=bool(payload.get("auto_retry_attempted") or False),
        retry_successful=payload.get("retry_successful"),
        retry_count=int(payload.get("retry_count") or 0),
    )


def try_drain_fallback_to_db(*, config: ErrorLoggingConfig | None = None) -> int:
    """Move pending fallback records into ``thrive_error_log``.

    Returns the count of rows actually inserted. Idempotent — records whose
    ``(created_at, error_type, error_message)`` already exists in the table
    are skipped. Records missing any required NOT NULL field are dropped.
    On any failure (read, query, insert, commit, file deletion), returns 0
    and leaves the fallback files in place. Never raises.

    Intended to be invoked manually from the Errors admin UI after the DB
    has been confirmed reachable again — not called automatically.
    """
    try:
        cfg = config or ErrorLoggingConfig.from_streamlit()

        records = read_fallback_records(since=datetime.min, config=cfg)
        if not records:
            return 0

        candidates: list[tuple[dict, datetime]] = []
        for record in records:
            if not all(record.get(k) for k in _REQUIRED_FOR_DB):
                continue
            try:
                when = datetime.fromisoformat(str(record["created_at"]))
            except (TypeError, ValueError):
                continue
            candidates.append((record, when))

        if not candidates:
            return 0

        with SessionLocal() as session:
            clauses = [
                and_(
                    ErrorLog.created_at == when,
                    ErrorLog.error_type == rec.get("error_type"),
                    ErrorLog.error_message == rec.get("error_message"),
                )
                for rec, when in candidates
            ]
            existing_keys: set[tuple] = set(
                session.query(
                    ErrorLog.created_at,
                    ErrorLog.error_type,
                    ErrorLog.error_message,
                )
                .filter(or_(*clauses))
                .all()
            )

            new_rows: list[ErrorLog] = []
            for rec, when in candidates:
                key = (when, rec["error_type"], rec["error_message"])
                if key in existing_keys:
                    continue
                new_rows.append(_build_error_log_row(rec, when))
                existing_keys.add(key)  # guard against duplicates within the batch

            if not new_rows:
                # Nothing new to write, but files contained only dupes — safe to clear.
                inserted = 0
            else:
                session.add_all(new_rows)
                session.commit()
                inserted = len(new_rows)

        # Close cached handlers (so file handles release) then delete sources.
        _reset_handler_cache()
        base = cfg.fallback_path
        paths = [base] + [base.with_name(f"{base.name}.{i}") for i in range(1, cfg.fallback_backup_count + 1)]
        for path in paths:
            try:
                if path.exists():
                    path.unlink()
            except OSError:
                pass

        return inserted
    except Exception:
        return 0
