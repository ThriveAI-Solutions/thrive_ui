"""SQL template UNIONing imaging-related rows.

Per spec §7.12: federated_orders_v (modality/body region from text/code)
UNION federated_documents_v (radiology report metadata only).

Impression text is NOT in the warehouse — callers must surface that via
notes_to_agent in the tool layer.

Modality matching is keyword-based against the order/document text. A
real implementation would prefer a dim_imaging_modality crosswalk; that
is tracked as future work in spec §14.

Body-region matching follows the same keyword pattern as modality: a
canonical region name (e.g. "chest") is expanded to a tuple of synonyms
in _BODY_REGION_KEYWORDS, each applied as LOWER()+LIKE against the order
name and document name/mnemonic columns. Unknown region names fall back to
a raw LIKE on the supplied string.
"""

from __future__ import annotations
from typing import Optional, Tuple


_MODALITY_KEYWORDS = {
    "xray": ("x-ray", "xray", "x ray", "radiograph"),
    "ct": ("ct ", " ct,", " ct;", "computed tomography", " ct scan"),
    "mri": ("mri", "magnetic resonance"),
    "us": ("ultrasound", " us ", "sonogram"),
    "pet": ("pet ", "positron emission"),
}

_BODY_REGION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "head": ("head", "brain", "cranial", "cranium", "skull", "intracranial"),
    "neck": ("neck", "cervical spine", "c-spine", "c spine", "thyroid"),
    "chest": ("chest", "thorax", "lung", "pulmonary", "cardiac", "rib"),
    "abdomen": ("abdomen", "abdominal", "liver", "kidney", "renal", "gallbladder", "pancrea"),
    "pelvis": ("pelvis", "pelvic", "bladder", "prostate", "uterus", "ovary", "ovarian"),
    "spine": ("spine", "spinal", "lumbar", "thoracic spine", "sacral", "vertebr"),
    "shoulder": ("shoulder", "rotator cuff", "scapula", "acromioclavicular"),
    "knee": ("knee", "patella", "patellar", "tibial plateau"),
    "hip": ("hip", "femoral head", "acetabul"),
    "ankle": ("ankle", "malleol", "talar"),
    "wrist": ("wrist", "carpal", "scaphoid", "distal radius"),
    "hand": ("hand", "finger", "metacarp", "phalanx", "phalanges"),
    "foot": ("foot", "toe", "metatars", "calcaneal", "plantar"),
    "extremity": ("extremity", "extremities", "limb", "upper extremity", "lower extremity"),
}


def imaging_sql(
    *,
    source_id: str,
    modality: Optional[str] = None,
    body_region: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    schema_prefix: str = "",
) -> Tuple[str, dict]:
    params: dict = {"source_id": source_id}

    modality_clause_orders = ""
    modality_clause_docs = ""
    if modality and modality != "any":
        keywords = _MODALITY_KEYWORDS.get(modality, ())
        if keywords:
            order_terms: list[str] = []
            doc_terms: list[str] = []
            for i, kw in enumerate(keywords):
                key = f"mod_{i}"
                params[key] = f"%{kw}%"
                order_terms.append(f"LOWER(name) LIKE :{key}")
                doc_terms.append(f"LOWER(name) LIKE :{key} OR LOWER(mnemonic) LIKE :{key}")
            modality_clause_orders = f"AND ({' OR '.join(order_terms)})"
            modality_clause_docs = f"AND ({' OR '.join(doc_terms)})"

    body_clause_orders = ""
    body_clause_docs = ""
    if body_region:
        region_key = body_region.lower().strip()
        keywords = _BODY_REGION_KEYWORDS.get(region_key)
        if keywords:
            order_terms: list[str] = []
            doc_terms: list[str] = []
            for j, kw in enumerate(keywords):
                key = f"body_{j}"
                params[key] = f"%{kw}%"
                order_terms.append(f"LOWER(name) LIKE :{key}")
                doc_terms.append(f"LOWER(name) LIKE :{key} OR LOWER(mnemonic) LIKE :{key}")
            body_clause_orders = f"AND ({' OR '.join(order_terms)})"
            body_clause_docs = f"AND ({' OR '.join(doc_terms)})"
        else:
            # Fallback: use the raw body_region string as a single LIKE pattern
            body_clause_orders = "AND LOWER(name) LIKE :body"
            body_clause_docs = "AND (LOWER(name) LIKE :body OR LOWER(mnemonic) LIKE :body)"
            params["body"] = f"%{region_key}%"

    date_filter_orders = ""
    date_filter_docs = ""
    if start_date:
        date_filter_orders += " AND date_of_procedure >= :start_date"
        date_filter_docs += " AND datetime >= :start_date"
        params["start_date"] = start_date
    if end_date:
        date_filter_orders += " AND date_of_procedure <= :end_date"
        date_filter_docs += " AND datetime <= :end_date"
        params["end_date"] = end_date

    sql = f"""
        SELECT
            'orders' AS source,
            source_id,
            code,
            code_type,
            name AS description,
            NULL AS mnemonic,
            date_of_procedure AS event_date,
            place_of_service,
            NULL AS location_name
        FROM {schema_prefix}federated_orders_v
        WHERE source_id = :source_id
          AND (
              LOWER(name) LIKE '%imag%'
              OR LOWER(name) LIKE '%x-ray%' OR LOWER(name) LIKE '%xray%'
              OR LOWER(name) LIKE '%ct %' OR LOWER(name) LIKE '%mri%'
              OR LOWER(name) LIKE '%ultrasound%' OR LOWER(name) LIKE '%sonogram%'
              OR LOWER(name) LIKE '%radiograph%' OR LOWER(name) LIKE '%pet %'
          )
          {modality_clause_orders}
          {body_clause_orders}
          {date_filter_orders}

        UNION ALL

        SELECT
            'documents' AS source,
            source_id,
            NULL AS code,
            NULL AS code_type,
            name AS description,
            mnemonic,
            datetime AS event_date,
            place_of_service,
            location_name
        FROM {schema_prefix}federated_documents_v
        WHERE source_id = :source_id
          AND (
              LOWER(name) LIKE '%radiolog%'
              OR LOWER(name) LIKE '%imag%'
              OR LOWER(mnemonic) LIKE '%xr%'
              OR LOWER(mnemonic) LIKE '%ct%'
              OR LOWER(mnemonic) LIKE '%mri%'
          )
          {modality_clause_docs}
          {body_clause_docs}
          {date_filter_docs}

        ORDER BY event_date DESC
    """
    return sql, params
