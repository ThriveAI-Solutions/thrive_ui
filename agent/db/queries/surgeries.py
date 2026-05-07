"""SQL template for invasive procedures / surgeries.

Filters the three procedure sources to surgical/invasive codes only and
LEFT JOINs federated_encounters_v to resolve performing provider name.

Orders branch: CPT surgery range 10004-69990.
Problems branch: invasive ICD-10-PCS root operations (3rd character).
Claims branch: broadly inclusive, excludes Inspection (root op J).
"""

from __future__ import annotations
from typing import List, Optional, Tuple

from agent.code_normalizer import variants_for


# ICD-10-PCS 3rd-character (root operation) values considered invasive.
_INVASIVE_ROOT_OPS = (
    "5",  # Destruction
    "8",  # Division
    "9",  # Drainage
    "B",  # Excision
    "C",  # Extirpation
    "D",  # Extraction
    "F",  # Fragmentation
    "G",  # Fusion
    "H",  # Insertion
    "N",  # Release
    "P",  # Removal
    "Q",  # Repair
    "R",  # Replacement
    "S",  # Reposition
    "T",  # Resection
    "U",  # Supplement
    "V",  # Restriction
    "W",  # Revision
    "X",  # Transfer
    "Y",  # Transplantation
)


def surgeries_sql(
    *,
    source_id: str,
    cpt_codes: Optional[List[str]] = None,
    procedure_text: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    schema_prefix: str = "",
    dialect: str = "sqlite",
) -> Tuple[str, dict]:
    params: dict = {"source_id": source_id}

    # --- Provider aggregation (dialect-aware) ---
    # When multiple encounters match the same day, join all distinct provider
    # names rather than silently picking one. The tool handler can detect
    # ambiguity by checking for the separator in the returned string.
    if dialect in ("postgres", "redshift"):
        provider_agg = "LISTAGG(DISTINCT e.rendering_provider, ', ') WITHIN GROUP (ORDER BY e.rendering_provider)"
    else:
        provider_agg = "GROUP_CONCAT(DISTINCT e.rendering_provider)"

    # --- CPT code filter (orders branch only) ---
    cpt_filter = ""
    icdpcs_filter = ""
    if cpt_codes:
        placeholders = ", ".join(f":cpt_{i}" for i in range(len(cpt_codes)))
        cpt_filter = f"AND o.code IN ({placeholders})"
        icdpcs_filter = "AND 1 = 0"
        for i, c in enumerate(cpt_codes):
            params[f"cpt_{i}"] = c

    # --- Text filters ---
    text_filter_orders = ""
    text_filter_problems = ""
    text_filter_claims = ""
    if procedure_text:
        text_filter_orders = "AND LOWER(o.name) LIKE :pt"
        text_filter_problems = "AND LOWER(p.diagnosis) LIKE :pt"
        text_filter_claims = "AND LOWER(c.procedure_description) LIKE :pt"
        params["pt"] = f"%{procedure_text.lower()}%"

    # --- Date filters ---
    date_filter_orders = ""
    date_filter_problems = ""
    date_filter_claims = ""
    if start_date:
        date_filter_orders += " AND o.date_of_procedure >= :start_date"
        date_filter_problems += " AND p.diagnosis_datetime >= :start_date"
        date_filter_claims += " AND c.procedure_date >= :start_date"
        params["start_date"] = start_date
    if end_date:
        date_filter_orders += " AND o.date_of_procedure <= :end_date"
        date_filter_problems += " AND p.diagnosis_datetime <= :end_date"
        date_filter_claims += " AND c.procedure_date <= :end_date"
        params["end_date"] = end_date

    # --- ICD-10-PCS code_type variants ---
    icd10pcs_variants = ("ICD-10-PCS", "ICD10-PCS")
    pcs_placeholders = ", ".join(f":pcs_{i}" for i in range(len(icd10pcs_variants)))
    for i, v in enumerate(icd10pcs_variants):
        params[f"pcs_{i}"] = v

    # --- CPT code_type variants ---
    cpt_variants = variants_for("cpt")
    cpt_ct_placeholders = ", ".join(f":cct_{i}" for i in range(len(cpt_variants)))
    for i, v in enumerate(cpt_variants):
        params[f"cct_{i}"] = v

    # --- Invasive root operations for problems branch ---
    inv_placeholders = ", ".join(f":inv_{i}" for i in range(len(_INVASIVE_ROOT_OPS)))
    for i, op in enumerate(_INVASIVE_ROOT_OPS):
        params[f"inv_{i}"] = op

    sql = f"""
        SELECT
            'orders' AS source,
            o.source_id,
            o.code,
            o.code_type,
            o.name AS description,
            o.date_of_procedure AS event_date,
            o.place_of_service,
            NULL AS provider_npi,
            NULL AS facility_name,
            {provider_agg} AS performing_provider
        FROM {schema_prefix}federated_orders_v o
        LEFT JOIN {schema_prefix}federated_encounters_v e
          ON o.source_id = e.source_id
         AND DATE(o.date_of_procedure) = DATE(e.datetime)
        WHERE o.source_id = :source_id
          AND o.code_type IN ({cpt_ct_placeholders})
          AND LENGTH(o.code) = 5
          AND o.code >= '10004' AND o.code <= '69990'
          {cpt_filter}
          {text_filter_orders}
          {date_filter_orders}
        GROUP BY o.source_id, o.code, o.code_type, o.name,
                 o.date_of_procedure, o.place_of_service

        UNION ALL

        SELECT
            'problems' AS source,
            p.source_id,
            p.code,
            p.code_type,
            p.diagnosis AS description,
            p.diagnosis_datetime AS event_date,
            NULL AS place_of_service,
            p.service_provider_npi AS provider_npi,
            NULL AS facility_name,
            {provider_agg} AS performing_provider
        FROM {schema_prefix}federated_problems_v p
        LEFT JOIN {schema_prefix}federated_encounters_v e
          ON p.source_id = e.source_id
         AND DATE(p.diagnosis_datetime) = DATE(e.datetime)
        WHERE p.source_id = :source_id
          AND p.code_type IN ({pcs_placeholders})
          AND SUBSTR(p.code, 3, 1) IN ({inv_placeholders})
          {icdpcs_filter}
          {text_filter_problems}
          {date_filter_problems}
        GROUP BY p.source_id, p.code, p.code_type, p.diagnosis,
                 p.diagnosis_datetime, p.service_provider_npi

        UNION ALL

        SELECT
            'claims' AS source,
            c.source_id,
            c.icd_procedure_code AS code,
            c.code_type,
            c.procedure_description AS description,
            c.procedure_date AS event_date,
            c.place_of_service,
            c.rendering_provider_npi AS provider_npi,
            c.facility_name,
            {provider_agg} AS performing_provider
        FROM {schema_prefix}federated_claims_icd_procedure_detail_v c
        LEFT JOIN {schema_prefix}federated_encounters_v e
          ON c.source_id = e.source_id
         AND DATE(c.procedure_date) = DATE(e.datetime)
        WHERE c.source_id = :source_id
          AND c.code_type IN ({pcs_placeholders})
          AND SUBSTR(c.icd_procedure_code, 3, 1) != 'J'
          {icdpcs_filter}
          {text_filter_claims}
          {date_filter_claims}
        GROUP BY c.source_id, c.icd_procedure_code, c.code_type,
                 c.procedure_description, c.procedure_date,
                 c.place_of_service, c.rendering_provider_npi,
                 c.facility_name

        ORDER BY event_date DESC
    """
    return sql, params
