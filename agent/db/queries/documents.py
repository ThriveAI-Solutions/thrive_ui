"""SQL template for federated_documents_v.

Per spec §7.3: index only — note bodies are not stored. The tool layer
must surface that note bodies must be retrieved from HEALTHeLINK / EHR.
"""

from __future__ import annotations
from typing import Optional, Tuple


def documents_sql(
    *,
    source_id: str,
    document_type: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    schema_prefix: str = "",
) -> Tuple[str, dict]:
    where: list[str] = ["source_id = :source_id"]
    params: dict = {"source_id": source_id}

    if document_type:
        where.append("LOWER(name) LIKE :dt")
        params["dt"] = f"%{document_type.lower()}%"

    if start_date:
        where.append("datetime >= :start_date")
        params["start_date"] = start_date
    if end_date:
        where.append("datetime <= :end_date")
        params["end_date"] = end_date

    sql = f"""
        SELECT
            source_id,
            datetime AS event_datetime,
            name,
            mnemonic,
            status,
            encounter_id,
            place_of_service,
            location_name
        FROM {schema_prefix}federated_documents_v
        WHERE {" AND ".join(where)}
        ORDER BY datetime DESC
    """
    return sql, params
