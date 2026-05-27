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
    schema_prefix: str = "",
) -> Tuple[str, dict]:
    params: dict = {"source_id": source_id}

    cpt_filter = ""
    # Suppresses the ICD-10-PCS path on the problems branch when the caller
    # is filtering by CPT codes (CPT and PCS code systems don't overlap, so
    # the PCS rows would just be noise alongside the requested CPT rows).
    problems_pcs_filter = ""
    # The claims branch is unconditionally suppressed in Phase 3 — see the
    # WHERE-clause comment near the FROM clause below for the patient-scoping
    # gap. Will become a real predicate when a JOIN bridge is wired up.
    claims_filter = "AND 1 = 0"
    if cpt_codes:
        placeholders = ", ".join(f":cpt_{i}" for i in range(len(cpt_codes)))
        cpt_filter = f"AND code IN ({placeholders})"
        problems_pcs_filter = "AND 1 = 0"
        for i, c in enumerate(cpt_codes):
            params[f"cpt_{i}"] = c

    text_filter_orders = ""
    text_filter_problems = ""
    # federated_claims_icd_procedure_detail_v has no procedure_description column
    # per redshift_tables.json; text filtering is not possible on the claims branch.
    if procedure_text:
        text_filter_orders = "AND LOWER(name) LIKE :pt"
        text_filter_problems = "AND LOWER(diagnosis) LIKE :pt"
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

    # ICD-10-PCS is the procedural-codes subset of ICD-10. Per spec §7.12,
    # only ~7% of problems rows are PCS-coded; non-PCS ICD-10 rows are
    # diagnoses (e.g. E11.9 diabetes) and must be excluded.
    icd10pcs_variants = ("ICD-10-PCS", "ICD10-PCS")
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
        FROM {schema_prefix}federated_orders_v
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
        FROM {schema_prefix}federated_problems_v
        WHERE source_id = :source_id
          AND code_type IN ({pcs_placeholders})
          {problems_pcs_filter}
          {text_filter_problems}
          {date_filter_problems}

        UNION ALL

        SELECT
            'claims' AS source,
            -- federated_claims_icd_procedure_detail_v has no source_id column per
            -- redshift_tables.json; NULL placeholder preserves UNION shape.
            NULL AS source_id,
            icd_procedure_code AS code,
            -- code_type does not exist; icd_type is the real column name per
            -- redshift_tables.json.
            icd_type AS code_type,
            -- procedure_description, place_of_service, rendering_provider_npi,
            -- and facility_name do not exist on this view per redshift_tables.json.
            NULL AS description,
            procedure_date AS event_date,
            NULL AS place_of_service,
            NULL AS provider_npi,
            NULL AS facility_name
        FROM {schema_prefix}federated_claims_icd_procedure_detail_v
        WHERE 1=1
          -- DEFERRED DEFECT (HIPAA-relevant): the claims procedure view has
          -- no patient identifier — not source_id, patient_id, umrn, or any
          -- direct join key. Its only foreign keys are claim_line_identifier
          -- and source_file_name. So this branch cannot be scoped to the
          -- requested source_id; without suppression it would return claims
          -- rows from EVERY patient in the date range.
          --
          -- Phase 3 hard-suppresses via claims_filter = "AND 1 = 0" (above).
          -- A real fix needs a 3-hop bridge:
          --   claims_icd_procedure_detail_v.claim_line_identifier
          --   → federated_claims_summary_v.claim_line_identifier
          --     (gives payer_member_identifier)
          --   → internal_source_reference_v WHERE source_name='claims_process'
          --     (assuming source_id stores payer_member_identifier; needs
          --      live-warehouse confirmation)
          --   → patient_id → source_id (empi_rank=1)
          -- Tracked as Phase 4 carry-over. Do not lift the suppression
          -- without that bridge in place AND a unit test asserting that
          -- ProceduresQuery for patient A never returns patient B's rows.
          {claims_filter}
          {date_filter_claims}

        ORDER BY event_date DESC
    """
    return sql, params
