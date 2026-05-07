"""SQL template for federated_meds_v.

Per spec §7.12: ndc_code and rxnorm_code are 100% populated. No
reliability badge needed. drug_class is not stored; callers wanting
class-level filters must supply rxnorm_codes themselves (Phase 4 cohort
work, not Phase 2).
"""

from __future__ import annotations
from typing import List, Optional, Tuple


def medications_sql(
    *,
    source_id: str,
    rxnorm_codes: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    schema_prefix: str = "",
) -> Tuple[str, dict]:
    where: list[str] = ["source_id = :source_id"]
    params: dict = {"source_id": source_id}

    if rxnorm_codes:
        placeholders = ", ".join(f":rx_{i}" for i in range(len(rxnorm_codes)))
        where.append(f"rxnorm_code IN ({placeholders})")
        for i, c in enumerate(rxnorm_codes):
            params[f"rx_{i}"] = c

    if start_date:
        where.append("date_prescribed >= :start_date")
        params["start_date"] = start_date
    if end_date:
        where.append("date_prescribed <= :end_date")
        params["end_date"] = end_date

    sql = f"""
        SELECT
            source_id,
            ndc_code,
            rxnorm_code,
            med_name,
            date_prescribed,
            prescribing_provider_npi,
            med_strength,
            med_strength_unit,
            med_form,
            med_sig,
            drug_supply_days,
            number_of_refills
        FROM {schema_prefix}federated_meds_v
        WHERE {" AND ".join(where)}
        ORDER BY date_prescribed DESC
    """
    return sql, params
