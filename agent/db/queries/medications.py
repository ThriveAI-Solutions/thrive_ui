"""SQL template for federated_meds_v.

Returns the patient's full medication list (optionally date-bounded). There
is no code-based filter: med_name is reliably populated and downstream
callers (the LLM) identify relevant drugs by name and status. status /
date_stopped distinguish active from discontinued prescriptions. Cohort-level
filtering lives in agent/db/queries/cohort.py, not here.
"""

from __future__ import annotations
from typing import Optional, Tuple


def medications_sql(
    *,
    source_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    schema_prefix: str = "",
) -> Tuple[str, dict]:
    where: list[str] = ["source_id = :source_id"]
    params: dict = {"source_id": source_id}

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
            number_of_refills,
            status,
            date_stopped
        FROM {schema_prefix}federated_meds_v
        WHERE {" AND ".join(where)}
        ORDER BY date_prescribed DESC
    """
    return sql, params
