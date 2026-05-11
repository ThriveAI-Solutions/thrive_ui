"""list_patient_documents — document INDEX (no bodies).

Per spec §7.3: returns metadata from federated_documents_v only. Note
bodies live in HEALTHeLINK / source EHRs and are not in this warehouse.
"""

from __future__ import annotations
from datetime import date
from typing import List, Literal, Optional

from pydantic import BaseModel
from pydantic_ai import ModelRetry, RunContext

from agent.deps import AgentDeps
from agent.db.queries.documents import documents_sql
from agent.dataframe_adapters import document_index_result_to_df


class DateRange(BaseModel):
    start: Optional[date] = None
    end: Optional[date] = None


class DocumentIndexQuery(BaseModel):
    document_type: Optional[str] = None
    date_range: Optional[DateRange] = None


class DocumentEntry(BaseModel):
    source_id: str
    event_datetime: Optional[str]
    name: Optional[str]
    mnemonic: Optional[str]
    status: Optional[str]
    encounter_id: Optional[str]
    place_of_service: Optional[str]
    location_name: Optional[str]


class DocumentIndexResult(BaseModel):
    documents: List[DocumentEntry]
    data_availability: Literal["data_present", "no_records_found", "error"]
    note: str = "Note bodies are not stored in this warehouse; retrieval requires HEALTHeLINK or source EHR access."


def list_patient_documents(
    ctx: RunContext[AgentDeps],
    query: DocumentIndexQuery,
) -> DocumentIndexResult:
    if ctx.deps.selected_patient is None:
        raise ModelRetry(
            "No patient is currently selected. Call find_patient first or ask the user to select a patient."
        )
    source_id = ctx.deps.selected_patient.source_id
    adapter = ctx.deps.analytics_db
    dr = query.date_range
    sql, params = documents_sql(
        source_id=source_id,
        document_type=query.document_type,
        start_date=dr.start.isoformat() if dr and dr.start else None,
        end_date=dr.end.isoformat() if dr and dr.end else None,
        schema_prefix=getattr(adapter, "schema_prefix", ""),
    )
    rows = adapter.fetch_all(sql, params)
    if not rows:
        result = DocumentIndexResult(documents=[], data_availability="no_records_found")
    else:
        entries = [
            DocumentEntry(
                source_id=r["source_id"],
                event_datetime=str(r["event_datetime"]) if r.get("event_datetime") else None,
                name=r.get("name"),
                mnemonic=r.get("mnemonic"),
                status=r.get("status"),
                encounter_id=r.get("encounter_id"),
                place_of_service=r.get("place_of_service"),
                location_name=r.get("location_name"),
            )
            for r in rows
        ]
        result = DocumentIndexResult(documents=entries, data_availability="data_present")

    ctx.deps.last_dataframe = document_index_result_to_df(result)
    return result
