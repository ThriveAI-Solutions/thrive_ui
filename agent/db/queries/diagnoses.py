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
    condition_text: Optional[str] = None,
    most_recent_only: bool = False,
) -> Tuple[str, dict]:
    where: list[str] = ["source_id = :source_id"]
    params: dict = {"source_id": source_id}

    if icd10_codes:
        ct_variants = variants_for("icd10")
        ct_placeholders = ", ".join(f":ct_{i}" for i in range(len(ct_variants)))
        where.append(f"code_type IN ({ct_placeholders})")
        for i, v in enumerate(ct_variants):
            params[f"ct_{i}"] = v

        code_placeholders = ", ".join(f":dc_{i}" for i in range(len(icd10_codes)))
        where.append(f"code IN ({code_placeholders})")
        for i, c in enumerate(icd10_codes):
            params[f"dc_{i}"] = c

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
            status_datetime,
            chronic_ind,
            service_provider_npi
        FROM federated_problems_v
        WHERE {where_sql}
        ORDER BY diagnosis_datetime DESC
    """

    if most_recent_only:
        base_sql += " LIMIT 1"

    return base_sql, params
