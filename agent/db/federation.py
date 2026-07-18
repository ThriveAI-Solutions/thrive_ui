"""EMPI federation helpers for patient-scoped domain queries.

A patient's chart is split across sibling source_ids (one per contributing
source system, linked via internal_source_reference_v.patient_id), so a
single-id filter silently drops most of the record — e.g. a prod patient
with every encounter under an empi_rank=2 sibling. Tools resolve the
selected source_id to the full federation set, query each sibling, and
merge. adt.py is the exception: it already resolves identity through the
EMPI join inside its SQL, so admissions must NOT go through this fan-out.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

from agent.db.queries.patient import all_source_ids_sql, resolve_source_patient_sql
from utils.quick_logger import get_logger

logger = get_logger(__name__)


def federated_source_ids(adapter: Any, source_id: str, schema_prefix: str = "") -> list[str]:
    """Resolve a source_id to the patient's full deduped federation set.

    internal_source_reference_v holds one row per (source_id, source_name),
    so the same source_id recurs across rows — dedupe is load-bearing.
    Ids unknown to EMPI (e.g. minimal test fixtures) fall back to the
    entered id alone.

    Ambiguity is fail-closed (#243): if the entered source_id resolves to more
    than one distinct internal patient, return an EMPTY set rather than
    silently picking one — federating one of two different real patients'
    charts would leak or misattribute data. Downstream retrieval then finds
    nothing, the same posture as a nonexistent id. Only a count is logged,
    never any identifier.
    """
    resolve_sql, _ = resolve_source_patient_sql(schema_prefix=schema_prefix)
    resolved = adapter.fetch_all(resolve_sql, {"source_id": source_id})
    if not resolved:
        return [source_id]
    patient_ids = {r["internal_patient_id"] for r in resolved}
    if len(patient_ids) > 1:
        logger.warning(
            "Ambiguous source_id resolves to %d distinct patients; refusing to federate.",
            len(patient_ids),
        )
        return []
    sid_sql, _ = all_source_ids_sql(schema_prefix=schema_prefix)
    sid_rows = adapter.fetch_all(sid_sql, {"internal_patient_id": next(iter(patient_ids))})
    sids = list(dict.fromkeys(r["source_id"] for r in sid_rows))
    return sids or [source_id]


def fetch_rows_across_source_ids(
    adapter: Any,
    source_ids: list[str],
    build_sql: Callable[[str], Tuple[str, dict]],
) -> list[dict]:
    """Run a per-source_id SQL builder for each sibling and concatenate rows."""
    rows: list[dict] = []
    for sid in source_ids:
        sql, params = build_sql(sid)
        rows.extend(adapter.fetch_all(sql, params))
    return rows


def sort_rows_date_desc(rows: list[dict], date_key: str) -> list[dict]:
    """Sort merged rows by event date descending with NULLs last.

    After merging, per-sid ORDER BY no longer gives a global order, and the
    LLM reads the first row as "most recent". str() normalizes date/datetime
    objects and ISO strings alike; within one domain column the format is
    uniform, so lexicographic order matches chronological order.
    """

    def key(row: dict) -> tuple[bool, str]:
        value: Optional[Any] = row.get(date_key)
        return (value is not None, str(value) if value is not None else "")

    return sorted(rows, key=key, reverse=True)
