"""SQL template for federated_problems_v (diagnoses).

Per spec §7.12: ICD-10 ~57%, SNOMED ~25%, ICD-9 ~5%. The icd10_codes
filter restricts code_type to ICD-10 variants (including ICD-10-CM /
ICD-10-PCS) and the canonical code list.
"""

from __future__ import annotations
from typing import List, Optional, Tuple

from agent.code_normalizer import variants_for


def diagnoses_sql(
    *,
    source_id: str,
    icd10_codes: Optional[List[str]] = None,
    condition_code_prefixes: Optional[List[str]] = None,
    condition_text: Optional[str] = None,
    most_recent_only: bool = False,
    schema_prefix: str = "",
) -> Tuple[str, dict]:
    where: list[str] = ["source_id = :source_id"]
    params: dict = {"source_id": source_id}

    # icd10_codes = exact codes; condition_code_prefixes = ICD-10 block prefixes
    # (e.g. ['E08','E11'] -> code LIKE 'E08%' OR 'E11%'), the NARROW reading of a
    # named condition. Both restrict code_type to ICD-10 variants and are OR'd.
    if icd10_codes or condition_code_prefixes:
        ct_variants = variants_for("icd10")
        ct_placeholders = ", ".join(f":ct_{i}" for i in range(len(ct_variants)))
        where.append(f"code_type IN ({ct_placeholders})")
        for i, v in enumerate(ct_variants):
            params[f"ct_{i}"] = v

        code_clauses: list[str] = []
        if icd10_codes:
            code_placeholders = ", ".join(f":dc_{i}" for i in range(len(icd10_codes)))
            code_clauses.append(f"code IN ({code_placeholders})")
            for i, c in enumerate(icd10_codes):
                params[f"dc_{i}"] = c
        if condition_code_prefixes:
            pref_clauses = []
            for i, p in enumerate(condition_code_prefixes):
                pref_clauses.append(f"code LIKE :cp_{i}")
                params[f"cp_{i}"] = f"{p}%"
            code_clauses.append("(" + " OR ".join(pref_clauses) + ")")
        # Single filter keeps the bare clause shape; only wrap when OR-combining
        # exact codes with prefix blocks.
        where.append(code_clauses[0] if len(code_clauses) == 1 else "(" + " OR ".join(code_clauses) + ")")

    if condition_text:
        where.append("LOWER(diagnosis) LIKE :ct")
        params["ct"] = f"%{condition_text.lower()}%"

    where_sql = " AND ".join(where)
    base_sql = f"""
        SELECT
            source_id,
            code,
            code_type,
            diagnosis,
            diagnosis_datetime,
            status,
            status_datetime,
            chronic_ind,
            service_provider_npi
        FROM {schema_prefix}federated_problems_v
        WHERE {where_sql}
        ORDER BY diagnosis_datetime DESC NULLS LAST
    """

    if most_recent_only:
        base_sql += " LIMIT 1"

    return base_sql, params
