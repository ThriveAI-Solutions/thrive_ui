"""SQL template for federated_vaccination_v.

Per spec §7.12: CVX is 100% populated. No reliability badge.
"""

from __future__ import annotations
from typing import List, Optional, Tuple


def immunizations_sql(
    *,
    source_id: str,
    cvx_codes: Optional[List[str]] = None,
    vaccine_text: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    schema_prefix: str = "",
) -> Tuple[str, dict]:
    where: list[str] = ["source_id = :source_id"]
    params: dict = {"source_id": source_id}

    if cvx_codes:
        placeholders = ", ".join(f":cvx_{i}" for i in range(len(cvx_codes)))
        where.append(f"cvx IN ({placeholders})")
        for i, c in enumerate(cvx_codes):
            params[f"cvx_{i}"] = c

    if vaccine_text:
        where.append("LOWER(vaccine) LIKE :vt")
        params["vt"] = f"%{vaccine_text.lower()}%"

    date_col = "COALESCE(status_datetime, datetime)"
    if start_date:
        where.append(f"{date_col} >= :start_date")
        params["start_date"] = start_date
    if end_date:
        where.append(f"{date_col} <= :end_date")
        params["end_date"] = end_date

    sql = f"""
        SELECT
            source_id,
            cvx,
            ndc,
            vaccine,
            manufacturer,
            lot,
            {date_col} AS event_datetime,
            location_name,
            mvx_code
        FROM {schema_prefix}federated_vaccination_v
        WHERE {" AND ".join(where)}
        ORDER BY {date_col} DESC
    """
    return sql, params
