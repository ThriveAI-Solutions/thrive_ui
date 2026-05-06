"""SQL template UNIONing the three procedure sources.

Per spec §7.12: federated_orders_v (CPT codes when code_type='CPT'),
federated_problems_v (ICD-10-PCS codes — anomalous ~7% subset), and
federated_claims_icd_procedure_detail_v (monthly refresh, longer lag).

Each branch carries a synthetic `source` column so the agent can
attribute provenance.
"""

from __future__ import annotations
from typing import List, Optional, Tuple

from agent.code_normalizer import variants_for


def procedures_sql(
    *,
    source_id: str,
    cpt_codes: Optional[List[str]] = None,
    procedure_text: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Tuple[str, dict]:
    params: dict = {"source_id": source_id}

    cpt_filter = ""
    icdpcs_filter = ""
    if cpt_codes:
        placeholders = ", ".join(f":cpt_{i}" for i in range(len(cpt_codes)))
        cpt_filter = f"AND code IN ({placeholders})"
        icdpcs_filter = "AND 1 = 0"
        for i, c in enumerate(cpt_codes):
            params[f"cpt_{i}"] = c

    text_filter_orders = ""
    text_filter_problems = ""
    text_filter_claims = ""
    if procedure_text:
        text_filter_orders = "AND LOWER(name) LIKE :pt"
        text_filter_problems = "AND LOWER(diagnosis) LIKE :pt"
        text_filter_claims = "AND LOWER(procedure_description) LIKE :pt"
        params["pt"] = f"%{procedure_text.lower()}%"

    date_filter_orders = ""
    date_filter_problems = ""
    date_filter_claims = ""
    if start_date:
        date_filter_orders += " AND date_of_procedure >= :start_date"
        date_filter_problems += " AND diagnosis_datetime >= :start_date"
        date_filter_claims += " AND procedure_date >= :start_date"
        params["start_date"] = start_date
    if end_date:
        date_filter_orders += " AND date_of_procedure <= :end_date"
        date_filter_problems += " AND diagnosis_datetime <= :end_date"
        date_filter_claims += " AND procedure_date <= :end_date"
        params["end_date"] = end_date

    icd10pcs_variants = variants_for("icd10")
    pcs_placeholders = ", ".join(f":pcs_{i}" for i in range(len(icd10pcs_variants)))
    for i, v in enumerate(icd10pcs_variants):
        params[f"pcs_{i}"] = v

    cpt_variants = variants_for("cpt")
    cpt_ct_placeholders = ", ".join(f":cct_{i}" for i in range(len(cpt_variants)))
    for i, v in enumerate(cpt_variants):
        params[f"cct_{i}"] = v

    sql = f"""
        SELECT
            'orders' AS source,
            source_id,
            code,
            code_type,
            name AS description,
            date_of_procedure AS event_date,
            place_of_service,
            NULL AS provider_npi,
            NULL AS facility_name
        FROM federated_orders_v
        WHERE source_id = :source_id
          AND (code_type IN ({cpt_ct_placeholders}) OR code_type = '' OR code_type IS NULL)
          {cpt_filter}
          {text_filter_orders}
          {date_filter_orders}

        UNION ALL

        SELECT
            'problems' AS source,
            source_id,
            code,
            code_type,
            diagnosis AS description,
            diagnosis_datetime AS event_date,
            NULL AS place_of_service,
            service_provider_npi AS provider_npi,
            NULL AS facility_name
        FROM federated_problems_v
        WHERE source_id = :source_id
          AND code_type IN ({pcs_placeholders})
          {icdpcs_filter}
          {text_filter_problems}
          {date_filter_problems}

        UNION ALL

        SELECT
            'claims' AS source,
            source_id,
            icd_procedure_code AS code,
            code_type,
            procedure_description AS description,
            procedure_date AS event_date,
            place_of_service,
            rendering_provider_npi AS provider_npi,
            facility_name
        FROM federated_claims_icd_procedure_detail_v
        WHERE source_id = :source_id
          {icdpcs_filter}
          {text_filter_claims}
          {date_filter_claims}

        ORDER BY event_date DESC
    """
    return sql, params
