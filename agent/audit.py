"""HIPAA-aware audit logging.

Per spec §8.7:
- arguments_json: verbatim for non-run_sql tools; hashed-predicate for run_sql
- result_summary: metadata only — row counts, data_availability, column names.
  Never DataFrame contents, never source_id, never names.
- DataFrames: NEVER persisted to audit. Logger raises if it sees one.
"""

from __future__ import annotations
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd
from sqlalchemy.orm import Session

from orm.models import ToolCall


_SQL_LITERAL_RE = re.compile(r"'((?:[^']|'')*)'")


def _hash_literal(literal: str) -> str:
    h = hashlib.sha256(literal.encode("utf-8")).hexdigest()[:12]
    return f"<HASH:{h}>"


def scrub_arguments_json(tool_name: str, args: dict) -> str:
    if tool_name == "run_sql":
        sql = args.get("sql", "")
        scrubbed_sql = _SQL_LITERAL_RE.sub(lambda m: f"'{_hash_literal(m.group(1))}'", sql)
        return json.dumps({**args, "sql": scrubbed_sql})
    return json.dumps(args, default=str)


def summarize_result(tool_name: str, result_obj: Any) -> str:
    if isinstance(result_obj, pd.DataFrame):
        raise ValueError(
            "Refusing to summarize a raw DataFrame — tools must wrap DataFrames in a typed result before audit."
        )

    if not isinstance(result_obj, dict):
        return f"result_type={type(result_obj).__name__}"

    parts: list[str] = []
    if "row_count" in result_obj:
        parts.append(f"row_count={result_obj['row_count']}")
    if "data_availability" in result_obj:
        parts.append(f"data_availability={result_obj['data_availability']}")
    if "domain" in result_obj:
        parts.append(f"domain={result_obj['domain']}")
    if "items" in result_obj and isinstance(result_obj["items"], list):
        parts.append(f"item_count={len(result_obj['items'])}")
    if "matches" in result_obj and isinstance(result_obj["matches"], list):
        parts.append(f"match_count={len(result_obj['matches'])}")
    if "total_unique" in result_obj:
        parts.append(f"total_unique={result_obj['total_unique']}")
    if "truncated" in result_obj:
        parts.append(f"truncated={result_obj['truncated']}")
    return "; ".join(parts) or "empty_result"


@dataclass
class AuditLogger:
    session: Session
    session_id: str
    user_id: int
    user_role: int

    def log(
        self,
        tool_name: str,
        selected_patient_source_id: Optional[str],
        arguments: dict,
        result_obj: Any,
        elapsed_ms: int,
        success: bool,
        error: Optional[str],
    ) -> None:
        row = ToolCall(
            session_id=self.session_id,
            user_id=self.user_id,
            user_role=self.user_role,
            selected_patient_source_id=selected_patient_source_id,
            tool_name=tool_name,
            arguments_json=scrub_arguments_json(tool_name, arguments),
            result_summary=summarize_result(tool_name, result_obj),
            elapsed_ms=elapsed_ms,
            success=success,
            error=error,
        )
        self.session.add(row)
        # Commit per call so we don't hold the SQLite write lock while the
        # agent yields more events. flush() alone leaves the transaction
        # open and blocks downstream Message.save() with "database is locked".
        self.session.commit()
