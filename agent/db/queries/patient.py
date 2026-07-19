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
    if not any((first_name, last_name, dob, mrn)):
        raise ValueError(
            "find_patient_sql requires at least one search criterion; refusing to issue an unfiltered scan."
        )

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

    where = " AND ".join(where_clauses)

    # Post-Fusion best-rank pick (#239): choose ONE source_id per patient via
    # ROW_NUMBER() PARTITION BY patient_id — lowest non-null empi_rank wins,
    # source_id as a deterministic tiebreaker. A hard `empi_rank = 1` filter is
    # NOT unique per patient: it both fans out duplicate result rows for
    # patients with several rank-1 xref rows and silently drops patients that
    # have no rank-1 row at all. empi_rank = 99 (stale) is excluded before the
    # pick. related_source_ids_sql returns the other ranks for the same patient.
    sql = f"""
    SELECT
        isr.source_id AS source_id,
        ipp.patient_id AS internal_patient_id,
        ipp.first_name AS first_name,
        ipp.last_name AS last_name,
        ipp.full_name AS display_name,
        ipp.date_of_birth AS dob,
        ipp.date_of_death AS date_of_death,
        ipp.age AS age,
        ipp.last_date_of_visit AS most_recent_activity,
        ipp.practice_name AS practice_name,
        isr.empi_rank AS empi_rank
    FROM {schema_prefix}internal_patient_profile_v ipp
    JOIN (
        SELECT patient_id, source_id, empi_rank FROM (
            SELECT patient_id, source_id, empi_rank,
                   ROW_NUMBER() OVER (PARTITION BY patient_id
                       ORDER BY COALESCE(empi_rank, 2147483647) ASC, source_id ASC
                   ) AS rn
            FROM {schema_prefix}internal_source_reference_v
            WHERE empi_rank <> 99
        ) ranked WHERE rn = 1
    ) isr ON ipp.patient_id = isr.patient_id
    WHERE {where}
    ORDER BY ipp.last_name, ipp.first_name, ipp.date_of_birth
    LIMIT :limit
    """
    return sql, params


def resolve_source_patient_sql(*, schema_prefix: str = "") -> Tuple[str, dict]:
    """Resolve an entered source_id to its internal patient_id(s).

    Returns the DISTINCT set of live patient_ids the source_id maps to — NOT a
    single ``LIMIT 1`` pick (#243). A source_id that maps to more than one
    patient is ambiguous and the caller must refuse rather than silently pick
    one, which could merge or misattribute two different patients' charts.
    Stale (empi_rank = 99) and the sentinel patient_id = -1 are excluded.
    Caller binds :source_id."""
    return (
        f"""
        SELECT DISTINCT patient_id AS internal_patient_id
        FROM {schema_prefix}internal_source_reference_v
        WHERE source_id = :source_id
          AND empi_rank <> 99
          AND patient_id <> -1
        """,
        {},
    )


def all_source_ids_sql(*, schema_prefix: str = "") -> Tuple[str, dict]:
    """All non-stale source_ids for an internal patient — INCLUDING the
    canonical (empi_rank = 1) row, so the full federation set is returned.
    Caller binds :internal_patient_id."""
    return (
        f"""
        SELECT source_id, empi_rank, source_name
        FROM {schema_prefix}internal_source_reference_v
        WHERE patient_id = :internal_patient_id
          AND empi_rank != 99
        ORDER BY empi_rank
        """,
        {},
    )


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
