"""Curated knowledge-base seed for the agent.

Per spec §10 Phase 1: RAG content covers ONLY the §7.12 view whitelist,
the §8.2 identity-column reality, and the §7.13 freshness disclosure.
Not all 103 views — just the focused set.
"""

from __future__ import annotations
from typing import List, TypedDict


class _Doc(TypedDict):
    view: str  # may be empty for non-view docs
    kind: str  # "schema" | "examples" | "docs"
    text: str


SCHEMA_DOCS: List[_Doc] = [
    {
        "view": "internal_patient_profile_v",
        "kind": "schema",
        "text": (
            "internal_patient_profile_v: consent-aware patient master. Use this as the "
            "primary source for find_patient. Key columns: patient_id (integer, "
            "warehouse-internal), first_name, last_name, full_name, date_of_birth, "
            "age, gender, last_date_of_visit, practice_name, provider_name, conditions. "
            "Patients in this view have given consent to be shown to other users; "
            "do NOT use federated_demographic_v for UI-visible patient lookup."
        ),
    },
    {
        "view": "internal_source_reference_v",
        "kind": "schema",
        "text": (
            "internal_source_reference_v: maps the warehouse-internal patient_id "
            "(integer) to canonical source_id (varchar). Filter empi_rank != 99 "
            "for active identities; rank 99 means inactive. One physical patient "
            "may have multiple source_ids across rank 1, 2, 3 — these are the same "
            "person under different source systems."
        ),
    },
    {
        "view": "federated_demographic_v",
        "kind": "schema",
        "text": (
            "federated_demographic_v: source-level demographic feed. NOT consent-filtered. "
            "Columns: source_id (varchar 50, canonical join key), patient_id (varchar, "
            "per-source — do not aggregate on this), date_of_birth, first_name, last_name, "
            "gender, address, phone, race, ethnicity. Use internal_patient_profile_v for "
            "patient lookup; use this view only when joined to a known source_id."
        ),
    },
    {
        "view": "federated_demographic_history_v",
        "kind": "schema",
        "text": (
            "federated_demographic_history_v: historical demographic snapshots, identical "
            "schema to federated_demographic_v."
        ),
    },
    {
        "view": "federated_encounters_v",
        "kind": "schema",
        "text": (
            "federated_encounters_v: encounter records. Columns: source_id, encounter_id, "
            "type, status, status_datetime (preferred timestamp — coalesced), datetime, "
            "location, rendering_provider, facility_name, place_of_service. "
            "place_of_service codes: 11=office, 21=inpatient, 22=outpatient, 23=ED, "
            "32/33/34/54/56=long-term care variants. NO discharge_date — admit only."
        ),
    },
    {
        "view": "federated_problems_v",
        "kind": "schema",
        "text": (
            "federated_problems_v: diagnoses and problems. Columns: source_id, code, "
            "code_type, diagnosis, diagnosis_datetime, status_datetime, chronic_ind, "
            "service_provider_npi. ICD-10 ~57%, SNOMED ~25%, ICD-9 legacy ~5%. "
            "code_type spelling varies — normalize via agent.code_normalizer."
        ),
    },
    {
        "view": "federated_results_v",
        "kind": "schema",
        "text": (
            "federated_results_v: lab results. Columns: source_id, code, code_type, "
            "name, mnemonic, result, clean_result, unit, datetime, service_provider. "
            "LOINC coverage ~50%; the rest are source-local codes. Always include "
            "a reliability caveat when answering from this view."
        ),
    },
    {
        "view": "federated_meds_v",
        "kind": "schema",
        "text": (
            "federated_meds_v: medications. Columns: source_id, ndc_code, rxnorm_code "
            "(both 100% populated), med_name, date_prescribed, prescribing_provider_npi, "
            "med_strength, med_strength_unit, med_form, med_sig, drug_supply_days, "
            "number_of_refills."
        ),
    },
    {
        "view": "federated_orders_v",
        "kind": "schema",
        "text": (
            "federated_orders_v: clinical orders. Columns: source_id, code, code_type, "
            "name, date_of_procedure, order_created_date, place_of_service, status. "
            "code_type empty ~46% — reliability caveat required. Useful for procedure "
            "and imaging metadata when CPT codes are present."
        ),
    },
    {
        "view": "federated_vaccination_v",
        "kind": "schema",
        "text": (
            "federated_vaccination_v: immunizations. Columns: source_id, cvx (100%), "
            "ndc, vaccine, manufacturer, lot, datetime, status_datetime, location_name, "
            "mvx_code. CVX is the canonical vaccine identifier; use search_codes "
            "with vocabulary='cvx' to look up codes from common names like 'MMR'."
        ),
    },
    {
        "view": "federated_vitals_v",
        "kind": "schema",
        "text": (
            "federated_vitals_v: vital signs. Columns: source_id, code, code_type "
            "(LOINC ~91%), name, result, clean_result, unit, datetime. Cleanest "
            "coverage of any clinical view."
        ),
    },
    {
        "view": "federated_documents_v",
        "kind": "schema",
        "text": (
            "federated_documents_v: document INDEX (no bodies). Columns: source_id, "
            "datetime, name (e.g., 'Progress Note'), mnemonic, status, encounter_id, "
            "place_of_service, location_name. Note bodies live in HEALTHeLINK / source "
            "EHRs, not the warehouse. Use list_patient_documents to surface what's "
            "available; never claim the agent has read the note text."
        ),
    },
    {
        "view": "metric_federated_data_v",
        "kind": "schema",
        "text": (
            "metric_federated_data_v: rolled-up event view spanning clinical + claims. "
            "Columns: patient_id (integer — join through internal_source_reference_v "
            "to get source_id), code, event_name, start_date, is_claims_data, "
            "data_source. Use when an answer needs to span domains without "
            "constructing UNIONs."
        ),
    },
]


IDENTITY_DOCS: List[_Doc] = [
    {
        "view": "",
        "kind": "schema",
        "text": (
            "Patient identity (CRITICAL): the canonical patient identifier is "
            "source_id (varchar). It is stable across insurance changes. "
            "There are TWO other patient_id columns: (1) federated_*_v.patient_id "
            "is varchar, per-source, changes with insurance — DO NOT aggregate on it. "
            "(2) internal_patient_profile_v.patient_id is integer, warehouse-internal "
            "— used only for joins to internal_source_reference_v. Always filter "
            "internal_source_reference_v with empi_rank != 99 to exclude inactive "
            "identities."
        ),
    },
]


FRESHNESS_DOCS: List[_Doc] = [
    {
        "view": "",
        "kind": "schema",
        "text": (
            "Data refresh cadence: federated clinical views refresh bi-weekly "
            "(twice per week). Claims tables refresh monthly. The system is not "
            "real-time. When answering questions about 'today', 'this week', or "
            "very recent events, add a freshness caveat: results may be up to "
            "4 days old (clinical) or up to a month old (claims/procedures)."
        ),
    },
]


EXAMPLES_DOCS: List[_Doc] = [
    {
        "view": "",
        "kind": "examples",
        "text": (
            "Q: 'Has John Smith had any negative Hepatitis results in the last year?'\n"
            "Tool sequence: find_patient(first_name='John', last_name='Smith') → "
            "[user picks one] → search_codes(vocabulary='loinc', query='hepatitis') → "
            "get_patient_clinical_data({domain:'labs', loinc_codes:[...], "
            "result_filter:'negative', date_range:{start:..., end:...}})."
        ),
    },
    {
        "view": "",
        "kind": "examples",
        "text": (
            "Q: 'Does this patient have a history of diabetes?'\n"
            "Tool sequence: get_patient_clinical_data({domain:'diagnoses', "
            "condition_text:'diabetes'}) → if positive, also "
            "get_patient_clinical_data({domain:'labs', "
            "test_name_text:'a1c', most_recent_only equivalent via date_range})."
        ),
    },
    {
        "view": "",
        "kind": "examples",
        "text": (
            "Q: 'Has patient ever had MMR vaccine?'\n"
            "Tool sequence: search_codes(vocabulary='cvx', query='mmr') → "
            "get_patient_clinical_data({domain:'immunizations', cvx_codes:['03']})."
        ),
    },
    {
        "view": "",
        "kind": "examples",
        "text": (
            "Q: 'Has patient been admitted to a long-term care facility in 2026?'\n"
            "Tool sequence: get_patient_clinical_data({domain:'encounters', "
            "facility_type:'ltc', date_range:{start:'2026-01-01', end:'2026-12-31'}})."
        ),
    },
    {
        "view": "",
        "kind": "examples",
        "text": (
            "Q: 'Did patient have imaging done last year?'\n"
            "Tool sequence: get_patient_clinical_data({domain:'imaging', "
            "date_range:{...}}). The result will set notes_to_agent reminding you "
            "that impression text is not stored — surface that to the user."
        ),
    },
    {
        "view": "",
        "kind": "examples",
        "text": (
            "Q: 'Has patient been treated with antibiotics for Gonorrhea after a date?'\n"
            "Tool sequence: search_codes(vocabulary='rxnorm', query='azithromycin') and "
            "search_codes(vocabulary='rxnorm', query='ceftriaxone') → "
            "get_patient_clinical_data({domain:'medications', rxnorm_codes:[...], "
            "date_range:{start:...}}). EPT and pregnancy-at-test are not in the warehouse — "
            "tell the user."
        ),
    },
    {
        "view": "",
        "kind": "examples",
        "text": (
            "Q: 'Does patient have a history of X disease?' (when the user is asking about "
            "documentation/notes)\n"
            "Tool sequence: list_patient_documents({document_type:'progress'}) and "
            "get_patient_clinical_data({domain:'diagnoses', condition_text:'X'}). "
            "Note bodies are not in the warehouse; surface that to the user."
        ),
    },
    {
        "view": "",
        "kind": "examples",
        "text": (
            "Q: 'Has this patient had any invasive procedures in date range?'\n"
            "Tool sequence: get_patient_clinical_data({domain:'procedures', "
            "date_range:{...}}). Result UNIONs orders, problems (ICD-10-PCS), and the "
            "claims feed; the claims feed lags by up to 30 days — include the freshness "
            "caveat from reliability_note."
        ),
    },
]


def all_seed_docs() -> List[_Doc]:
    return [*SCHEMA_DOCS, *IDENTITY_DOCS, *FRESHNESS_DOCS, *EXAMPLES_DOCS]
