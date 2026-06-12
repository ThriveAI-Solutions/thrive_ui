"""Resolve a roster source_id to a SelectedPatient via the warehouse.

source_id is the canonical patient identifier (spec §8.2); display name
and DOB come from internal_patient_profile_v, matching what the find_patient
tool would have surfaced.
"""

from __future__ import annotations

from datetime import date, datetime

from agent.deps import SelectedPatient

# Unlike find_patient_sql, no empi_rank filter: source_id is a unique row
# key in internal_source_reference_v, so direct lookup + LIMIT 1 suffices.
_RESOLVE_SQL = """
SELECT
    ipp.full_name AS display_name,
    ipp.date_of_birth AS dob
FROM {prefix}internal_source_reference_v isr
JOIN {prefix}internal_patient_profile_v ipp
  ON ipp.patient_id = isr.patient_id
WHERE isr.source_id = :source_id
LIMIT 1
"""


def _coerce_dob(raw) -> date | None:
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw.date()
    if isinstance(raw, date):
        return raw
    try:
        return date.fromisoformat(str(raw)[:10])
    except ValueError:
        return None


def resolve_patient(adapter, source_id: str) -> SelectedPatient:
    sql = _RESOLVE_SQL.format(prefix=adapter.schema_prefix)
    rows = adapter.fetch_all(sql, {"source_id": source_id})
    if not rows:
        raise LookupError(f"source_id {source_id!r} not found in warehouse")
    row = rows[0]
    return SelectedPatient(
        source_id=source_id,
        display_name=str(row.get("display_name") or source_id),
        dob=_coerce_dob(row.get("dob")),
        selected_at=datetime.now(),
        selection_origin="user_click",
    )
