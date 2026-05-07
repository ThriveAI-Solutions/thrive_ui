"""SQL templates for patient lookup.

Per spec §8.2: identity = source_id (varchar). Join through
internal_source_reference_v and pick the canonical (empi_rank = 1)
source_id per patient — this is how we deduplicate across feeds.
``related_source_ids_sql`` returns the other non-99 ranks for the same
internal patient. Use internal_patient_profile_v as the consent-aware
demographic source.
"""

from __future__ import annotations
from typing import Optional, Tuple


def find_patient_sql(
    *,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
    dob: Optional[str] = None,
    mrn: Optional[str] = None,
    limit: int = 25,
    schema_prefix: str = "",
) -> Tuple[str, dict]:
    where_clauses: list[str] = []
    params: dict = {"limit": limit}

    if first_name:
        where_clauses.append("LOWER(ipp.first_name) LIKE :first_name")
        params["first_name"] = f"%{first_name.lower()}%"
    if last_name:
        where_clauses.append("LOWER(ipp.last_name) LIKE :last_name")
        params["last_name"] = f"%{last_name.lower()}%"
    if dob:
        where_clauses.append("ipp.date_of_birth = :dob")
        params["dob"] = dob
    if mrn:
        where_clauses.append("ipp.umrn = :mrn")
        params["mrn"] = mrn

    where = " AND ".join(where_clauses) if where_clauses else "1=1"

    # isr.empi_rank = 1 picks the canonical source_id per patient (one
    # row per patient). 99 is automatically excluded since 99 != 1.
    # related_source_ids_sql returns the other ranks for the same
    # internal patient.
    sql = f"""
    SELECT
        isr.source_id AS source_id,
        ipp.patient_id AS internal_patient_id,
        ipp.first_name AS first_name,
        ipp.last_name AS last_name,
        ipp.full_name AS display_name,
        ipp.date_of_birth AS dob,
        ipp.age AS age,
        ipp.last_date_of_visit AS most_recent_activity,
        ipp.practice_name AS practice_name,
        isr.empi_rank AS empi_rank
    FROM {schema_prefix}internal_patient_profile_v ipp
    JOIN {schema_prefix}internal_source_reference_v isr
      ON ipp.patient_id = isr.patient_id
     AND isr.empi_rank = 1
    WHERE {where}
    ORDER BY ipp.last_name, ipp.first_name, ipp.date_of_birth
    LIMIT :limit
    """
    return sql, params


def related_source_ids_sql(*, schema_prefix: str = "") -> Tuple[str, dict]:
    return (
        f"""
        SELECT source_id, empi_rank, source_name
        FROM {schema_prefix}internal_source_reference_v
        WHERE patient_id = :internal_patient_id
          AND empi_rank != 99
          AND empi_rank != 1
        ORDER BY empi_rank
        """,
        {},
    )
