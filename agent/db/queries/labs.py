"""SQL template for federated_results_v (lab results).

Per spec §7.12: LOINC coverage ~50%; reliability badge required when
returning rows whose code_type is not a LOINC variant.
"""

from __future__ import annotations
from typing import List, Optional, Tuple

from agent.code_normalizer import variants_for


_RESULT_FILTER_MAP = {
    "negative": ("clean_result LIKE :rf_neg", {"rf_neg": "%negative%"}),
    "positive": ("clean_result LIKE :rf_pos", {"rf_pos": "%positive%"}),
    "abnormal": (
        "(clean_result LIKE :rf_abn OR clean_result LIKE :rf_h OR clean_result LIKE :rf_l)",
        {"rf_abn": "%abnormal%", "rf_h": "%high%", "rf_l": "%low%"},
    ),
}


def labs_sql(
    *,
    source_id: str,
    loinc_codes: Optional[List[str]] = None,
    test_name_text: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    result_filter: Optional[str] = None,
) -> Tuple[str, dict]:
    where: list[str] = ["source_id = :source_id"]
    params: dict = {"source_id": source_id}

    if loinc_codes:
        loinc_variants = variants_for("loinc")
        ct_placeholders = ", ".join(f":ct_{i}" for i in range(len(loinc_variants)))
        where.append(f"code_type IN ({ct_placeholders})")
        for i, v in enumerate(loinc_variants):
            params[f"ct_{i}"] = v

        code_placeholders = ", ".join(f":lc_{i}" for i in range(len(loinc_codes)))
        where.append(f"code IN ({code_placeholders})")
        for i, c in enumerate(loinc_codes):
            params[f"lc_{i}"] = c

    if test_name_text:
        where.append("LOWER(name) LIKE :tnt")
        params["tnt"] = f"%{test_name_text.lower()}%"

    if start_date:
        where.append("datetime >= :start_date")
        params["start_date"] = start_date
    if end_date:
        where.append("datetime <= :end_date")
        params["end_date"] = end_date

    if result_filter and result_filter != "any":
        clause = _RESULT_FILTER_MAP.get(result_filter)
        if clause:
            where.append(clause[0])
            params.update(clause[1])

    sql = f"""
        SELECT
            source_id,
            code,
            code_type,
            name,
            result,
            clean_result,
            unit,
            datetime AS event_datetime,
            service_provider
        FROM federated_results_v
        WHERE {" AND ".join(where)}
        ORDER BY datetime DESC
    """
    return sql, params
