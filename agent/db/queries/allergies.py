"""SQL template for federated_allergies_v.

Per epic #201: the dedicated allergies view exposes structured allergen,
category (`type`), and severity columns plus a 'NO KNOWN ALLERGIES'
negative-assertion row. Schema is inferred from the dev-warehouse
evaluation set + the federated_problems_v shape — verify columns against
production before merge.
"""

from __future__ import annotations
from typing import List, Optional, Tuple


# Maps the public `category` filter onto the warehouse `type` column.
# 'type' values seen in the eval set: 'Drug allergy', 'Food allergy',
# 'Adverse Reaction', 'None' (no-known-allergies rows).
_CATEGORY_TO_TYPES: dict[str, list[str]] = {
    "drug": ["Drug allergy"],
    "food": ["Food allergy"],
    "environmental": ["Environmental allergy"],
    "contact": ["Contact allergy"],
    "anaphylaxis": ["Adverse Reaction"],
}


def allergies_sql(
    *,
    source_id: str,
    snomed_codes: Optional[List[str]] = None,
    allergen_text: Optional[str] = None,
    category: Optional[str] = None,
    include_inactive: bool = False,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    schema_prefix: str = "",
) -> Tuple[str, dict]:
    where: list[str] = ["source_id = :source_id"]
    params: dict = {"source_id": source_id}

    if snomed_codes:
        placeholders = ", ".join(f":sn_{i}" for i in range(len(snomed_codes)))
        where.append(f"code IN ({placeholders})")
        for i, c in enumerate(snomed_codes):
            params[f"sn_{i}"] = c

    if allergen_text:
        where.append("LOWER(allergy) LIKE :at")
        params["at"] = f"%{allergen_text.lower()}%"

    if category:
        types = _CATEGORY_TO_TYPES.get(category.lower())
        if types:
            placeholders = ", ".join(f":ty_{i}" for i in range(len(types)))
            where.append(f"type IN ({placeholders})")
            for i, t in enumerate(types):
                params[f"ty_{i}"] = t

    if not include_inactive:
        # Active / unresolved allergies. The 'NO KNOWN ALLERGIES' row is
        # status='Active' by convention so it is preserved by this filter.
        where.append("(status IS NULL OR LOWER(status) IN ('active', 'unresolved'))")

    date_col = "COALESCE(status_datetime, onset_date)"
    if start_date:
        where.append(f"{date_col} >= :start_date")
        params["start_date"] = start_date
    if end_date:
        where.append(f"{date_col} <= :end_date")
        params["end_date"] = end_date

    sql = f"""
        SELECT
            source_id,
            code,
            code_type,
            allergy,
            type,
            severity,
            status,
            onset_date,
            {date_col} AS event_datetime,
            reaction,
            comments
        FROM {schema_prefix}federated_allergies_v
        WHERE {" AND ".join(where)}
        ORDER BY {date_col} DESC
    """
    return sql, params
