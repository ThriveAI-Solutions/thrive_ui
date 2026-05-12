"""find_patient tool — disambiguation entry point.

Per spec §7.1: returns deduped patients keyed on canonical source_id.
empi_rank != 99 enforced by SQL template (see agent/db/queries/patient.py).
"""

from __future__ import annotations
from datetime import date
from typing import List, Optional

from pydantic import BaseModel, Field
from pydantic_ai import RunContext

from agent.deps import AgentDeps
from agent.db.queries.patient import find_patient_sql, related_source_ids_sql
from agent.result_compaction import CompactingListResult


class PatientSearchQuery(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    dob: Optional[date] = None
    mrn: Optional[str] = None
    limit: int = Field(default=25, le=100)


class PatientMatch(BaseModel):
    source_id: str
    internal_patient_id: int
    related_source_ids: List[str]
    display_name: str
    dob: Optional[date]
    age: Optional[int]
    facilities_seen: List[str]
    record_count: int
    most_recent_activity: Optional[date]


class PatientSearchResults(CompactingListResult):
    _list_field = "matches"

    matches: List[PatientMatch]
    total_unique: int
    truncated: bool


def find_patient(
    ctx: RunContext[AgentDeps],
    query: PatientSearchQuery,
) -> PatientSearchResults:
    """Find patients by name, DOB, or MRN. Always dedupes by source_id."""
    adapter = ctx.deps.analytics_db
    schema_prefix = getattr(adapter, "schema_prefix", "")
    sql, params = find_patient_sql(
        first_name=query.first_name,
        last_name=query.last_name,
        dob=query.dob.isoformat() if query.dob else None,
        mrn=query.mrn,
        limit=query.limit + 1,
        schema_prefix=schema_prefix,
    )
    rows = adapter.fetch_all(sql, params)
    truncated = len(rows) > query.limit
    rows = rows[: query.limit]

    related_sql, _ = related_source_ids_sql(schema_prefix=schema_prefix)

    matches: List[PatientMatch] = []
    for r in rows:
        related = adapter.fetch_all(related_sql, {"internal_patient_id": r["internal_patient_id"]})
        matches.append(
            PatientMatch(
                source_id=r["source_id"],
                internal_patient_id=r["internal_patient_id"],
                related_source_ids=[x["source_id"] for x in related],
                display_name=r["display_name"] or f"{r['first_name']} {r['last_name']}",
                dob=r["dob"] if isinstance(r["dob"], date) else (date.fromisoformat(r["dob"]) if r["dob"] else None),
                age=r["age"],
                facilities_seen=[r["practice_name"]] if r.get("practice_name") else [],
                record_count=0,
                most_recent_activity=(
                    r["most_recent_activity"]
                    if isinstance(r.get("most_recent_activity"), date)
                    else (date.fromisoformat(r["most_recent_activity"]) if r.get("most_recent_activity") else None)
                ),
            )
        )

    return PatientSearchResults(
        matches=matches,
        total_unique=len(matches),
        truncated=truncated,
    )
