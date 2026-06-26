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
            "age, gender, last_date_of_visit, practice_name, provider_name, conditions, "
            "address_1, address_2, city, state, zip_code (5-digit, ~100% coverage). "
            "Patients in this view have given consent to be shown to other users; "
            "do NOT use federated_demographic_v for UI-visible patient lookup."
        ),
    },
    {
        "view": "internal_source_reference_v",
        "kind": "schema",
        "text": (
            "internal_source_reference_v: maps the warehouse-internal patient_id "
            "(integer) to canonical source_id (varchar). empi_rank: 1 = current "
            "primary CID (one per person); 2–10 = legacy/merged CIDs, same "
            "person; 99 = inactive. Count PEOPLE: COUNT(DISTINCT source_id) at "
            "empi_rank = 1. empi_rank != 99 returns ALL CIDs and over-counts "
            "people ~2x — use only to gather a person's facts."
        ),
    },
    {
        "view": "federated_demographic_v",
        "kind": "schema",
        "text": (
            "federated_demographic_v: source-level demographic feed. NOT consent-filtered. "
            "Columns: source_id (varchar 50, canonical join key), patient_id (varchar, "
            "per-source — do not aggregate on this), date_of_birth, first_name, last_name, "
            "gender, address_1, address_2, city, state, zip (no underscore on this view), "
            "phone, race, ethnicity. Use internal_patient_profile_v for "
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
        "view": "federated_adt_v",
        "kind": "schema",
        "text": (
            "federated_adt_v: admit/discharge/transfer events, queried via "
            "get_patient_clinical_data(domain='admissions') and rolled up to one row "
            "per visit_number (a stay). patient_id only (no source_id — join "
            "internal_source_reference_v at empi_rank=1). Each stay carries "
            "is_inpatient_admission: true iff a non-cancelled row has clean_setting "
            "INPATIENT or clean_status A06 (outpatient->inpatient), excluding pre-admit/"
            "pending (A05,A14,A38,A27) and any visit with CANCEL ADMIT. A bare ADMIT/A01 "
            "(often EMERGENCY-class) is NOT inpatient on its own. Supports facility_type "
            "(inpatient, ltc, snf, ed, outpatient, any) and date_range; inpatient "
            "date filters use the qualifying admission date."
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
        "view": "federated_allergies_v",
        "kind": "schema",
        "text": (
            "federated_allergies_v: allergy records. Columns: source_id, code "
            "(SNOMED), code_type, allergy, type (Drug/Food/Adverse Reaction/"
            "Environmental/Contact), severity, status, onset_date, status_datetime, "
            "reaction. Use get_patient_clinical_data domain='allergies'; NKA "
            "surfaces via the result's negative_assertion flag, drug-med conflicts "
            "via notes_to_agent. Uncoded allergies in notes are NOT captured."
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
            "source_id (varchar), stable across insurance changes. "
            "There are TWO other patient_id columns: (1) federated_*_v.patient_id "
            "is varchar, per-source, changes with insurance — DO NOT aggregate on it. "
            "(2) internal_patient_profile_v.patient_id is integer, warehouse-internal "
            "— used only for joins to internal_source_reference_v. To count distinct "
            "PEOPLE, COUNT(DISTINCT source_id) at empi_rank = 1 (current primary "
            "CID); empi_rank != 99 over-counts people."
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
            "Q: 'What is this patient allergic to?'\n"
            "Tool sequence: get_patient_clinical_data({domain:'allergies'}). "
            "Default returns active allergies. If the result has "
            "negative_assertion=True, the patient has explicitly asserted NO "
            "KNOWN ALLERGIES — say that, do not say 'I don't know'. If "
            "notes_to_agent contains a drug-allergy advisory, surface it as a "
            "soft flag for clinician review (NOT clinical decision support)."
        ),
    },
    {
        "view": "",
        "kind": "examples",
        "text": (
            "Q: 'Is it safe to give this patient penicillin?'\n"
            "Tool sequence: search_codes(vocabulary='snomed', query='penicillin "
            "allergy') → get_patient_clinical_data({domain:'allergies', "
            "snomed_codes:['91936005']}). Note: the agent reports recorded "
            "allergens only; coverage is limited to allergies captured in "
            "federated_allergies_v. Do not present a clinical safety verdict."
        ),
    },
    {
        "view": "",
        "kind": "examples",
        "text": (
            "Q: 'Has patient been admitted to a long-term care facility in 2026?'\n"
            "Tool sequence: get_patient_clinical_data({domain:'admissions', "
            "facility_type:'ltc', date_range:{start:'2026-01-01', end:'2026-12-31'}}). "
            "The admissions domain queries federated_adt_v for ADT events filtered by "
            "facility type and date range. For visit history without ADT detail, use "
            "get_patient_clinical_data({domain:'encounters', facility_type:'ltc', "
            "date_range:{...}}) instead."
        ),
    },
    {
        "view": "",
        "kind": "examples",
        "text": (
            "Q: 'What was the reason for patient X's most recent admission?'\n"
            "Tool sequence: (1) get_patient_clinical_data({domain:'admissions'}) — "
            "stays are ordered admit_date DESC, so items[0] is the most recent; "
            "(2) get_patient_clinical_data({domain:'diagnoses', date_range:{start:"
            "<admission_date - 1 day>, end:<admission_date + 1 day>}}) to surface "
            "diagnoses recorded around the admission. federated_adt_v has no "
            "principal-diagnosis column, so reason is INFERRED from problems near "
            "the admission date — say so in the answer. If "
            "data_availability='no_records_found', say explicitly that the patient "
            "has no admission records — do NOT phrase it as 'patient was not admitted'."
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
            "Tool sequence: get_patient_clinical_data({domain:'medications', "
            "date_range:{start:...}}) → scan the returned med_name values yourself "
            "for the relevant antibiotics (e.g. azithromycin, ceftriaxone); do not "
            "look up RxNorm codes for medications. EPT and pregnancy-at-test are not "
            "in the warehouse — tell the user."
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
            "date_range:{...}}). Result UNIONs orders (CPT) and problems "
            "(ICD-10-PCS only). The claims-feed branch is currently suppressed "
            "because the claims procedure view has no patient identifier — "
            "results may under-report procedures historically captured only "
            "via claims. Always surface the reliability_note from the result."
        ),
    },
    {
        "view": "",
        "kind": "examples",
        "text": (
            "Q: 'What is the most recent A1C value for this patient?'\n"
            "Tool sequence: search_codes(vocabulary='loinc', query='a1c') → "
            "get_patient_clinical_data({domain:'labs', loinc_codes:['4548-4'], "
            "most_recent_only:True}). The most_recent_only flag returns only "
            "the single most recent result ordered by datetime DESC."
        ),
    },
    {
        "view": "",
        "kind": "examples",
        "text": (
            "Q: 'Has this patient had Measles/Rubeola IgM/IgG testing done ever?'\n"
            "Tool sequence: search_codes(vocabulary='loinc', query='measles igg') → "
            "get_patient_clinical_data({domain:'labs', loinc_codes:['22501-7','22502-5']})."
        ),
    },
    {
        "view": "",
        "kind": "examples",
        "text": (
            "Q: 'Which facilities was this patient admitted to? What were the "
            "admission and discharge dates?'\n"
            "Tool sequence: get_patient_clinical_data({domain:'admissions', "
            "facility_type:'any'}). Returns one row per visit (a stay) with "
            "event_location, admit_date, discharge_date, discharge_disposition, and "
            "discharge_location."
        ),
    },
    {
        "view": "",
        "kind": "examples",
        "text": (
            "Q: 'Has patient ever had Hepatitis A, Strep Pneumo, Tdap, or Rabies vaccine?'\n"
            "Tool sequence: For each vaccine, call search_codes(vocabulary='cvx', "
            "query=<vaccine_name>) to get CVX codes, then call "
            "get_patient_clinical_data({domain:'immunizations', "
            "cvx_codes:[<all_codes>]})."
        ),
    },
    {
        "view": "",
        "kind": "examples",
        "text": (
            "Q: 'Did patient have imaging done on their knee last year? What type?'\n"
            "Tool sequence: get_patient_clinical_data({domain:'imaging', "
            "body_region:'knee', date_range:{start:..., end:...}}). "
            "body_region expands to case-insensitive keyword patterns covering 'knee', 'patella', "
            "'patellar', 'tibial plateau'."
        ),
    },
    {
        "view": "",
        "kind": "examples",
        "text": (
            "Q: 'Was this patient admitted to an inpatient facility in 2025?'\n"
            "Tool sequence: get_patient_clinical_data({domain:'admissions', "
            "facility_type:'inpatient', date_range:{start:'2025-01-01', end:'2025-12-31'}}). "
            "Returns one row per qualifying inpatient stay; answer yes iff any stay is "
            "returned (each has is_inpatient_admission=true). For all facility history "
            "(ED/SNF/outpatient too), omit facility_type and read is_inpatient_admission "
            "per stay."
        ),
    },
    {
        "view": "internal_patient_profile_v",
        "kind": "examples",
        "text": (
            "Q: 'How many patients over 65 with diabetes at Kaleida last year?' "
            "A: Call search_patients_by_criteria with diagnosis_codes=['E11.9'], "
            "age_min=65, facility='Kaleida', diagnosis_date_range covering the last year, "
            "sample_size=20. Surface the reliability_note (ICD-10 ~57% coverage)."
        ),
    },
    {
        "view": "internal_patient_profile_v",
        "kind": "examples",
        "text": (
            "Q: 'How many patients were admitted to an inpatient facility in 2025?' "
            "A: Call search_patients_by_criteria with inpatient_admission=true, "
            "inpatient_admission_date_range={start:'2025-01-01', end:'2025-12-31'}, "
            "sample_size=0; read total_count. This uses the inpatient-admission "
            "definition (clean_setting INPATIENT or A06 conversion, excluding "
            "pre-admit/pending and cancelled), not raw ADMIT events."
        ),
    },
    {
        "view": "internal_patient_profile_v",
        "kind": "examples",
        "text": (
            "Q: 'Find me 10 patients on metformin.' "
            "A: Call search_patients_by_criteria with medication_rxnorm_codes=['6809'], "
            "sample_size=10. Do NOT call find_patient first."
        ),
    },
    {
        "view": "internal_patient_profile_v",
        "kind": "examples",
        "text": (
            "Q: 'Active diabetic patients seen since 2025.' "
            "A: Call search_patients_by_criteria with diagnosis_codes=['E11.9'], "
            "last_visit_after=2025-01-01."
        ),
    },
    {
        "view": "internal_patient_profile_v",
        "kind": "examples",
        "text": (
            "Q: 'List people in zip 14223 with high blood pressure.' "
            "A: search_codes(vocabulary='icd10', query='hypertension') → "
            "search_patients_by_criteria(diagnosis_codes=['I10'], zip_code='14223'). "
            "Do NOT also pass condition_text — it AND-stacks against the code filter "
            "and returns zero. Surface the geo reliability note from the result."
        ),
    },
    {
        "view": "internal_patient_profile_v",
        "kind": "examples",
        "text": (
            "Q: 'How many people had any diagnosis in 2024? Break it out by month.' "
            "A: Call search_patients_by_criteria with "
            "diagnosis_date_range={start:'2024-01-01', end:'2024-12-31'} and "
            "breakdown=['diagnosis_month']. Do NOT write a run_sql GROUP BY. The "
            "result returns one bucket per month; surface the note that buckets "
            "count distinct active patients and OVERLAP (a patient diagnosed in "
            "two months is counted in both, so months do not sum to the total)."
        ),
    },
    {
        "view": "internal_patient_profile_v",
        "kind": "examples",
        "text": (
            "Q: 'Diabetic patients by gender.' "
            "A: search_patients_by_criteria(diagnosis_codes=['E11.9'], "
            "breakdown=['gender']). Gender is single-valued so buckets sum to the "
            "total. For TWO dimensions (e.g. month AND gender) the tool returns a "
            "generated_sql template to extend in run_sql."
        ),
    },
]


# Few-shot SQL templates for run_sql's tool description. The "{p}" token
# is replaced with the analytics_db.schema_prefix at render time
# (production "dw." / SQLite tests "") so the same source-of-truth
# example works across environments.
RUN_SQL_EXAMPLES: List[_Doc] = [
    {
        "view": "",
        "kind": "examples",
        "text": (
            "Q: List patients in a given zip with a given ICD-10 diagnosis.\n"
            "SQL:\n"
            "  SELECT isr.source_id, p.full_name, p.age, p.gender, p.last_date_of_visit\n"
            "  FROM {p}internal_patient_profile_v p\n"
            "  JOIN {p}internal_source_reference_v isr\n"
            "    ON isr.patient_id = p.patient_id AND isr.empi_rank = 1\n"
            "  JOIN (SELECT DISTINCT patient_id FROM {p}metric_federated_data_v\n"
            "        WHERE code = :icd10 AND code_type IN ('ICD-10','ICD10','SNOMED')) dx\n"
            "    ON dx.patient_id = p.patient_id\n"
            "  WHERE p.zip_code = :zip\n"
            "  LIMIT 100;"
        ),
    },
    {
        "view": "",
        "kind": "examples",
        "text": (
            "Q: Count distinct patients seen since a date, grouped by practice.\n"
            "SQL:\n"
            "  SELECT p.practice_name, COUNT(DISTINCT isr.source_id) AS patient_count\n"
            "  FROM {p}internal_patient_profile_v p\n"
            "  JOIN {p}internal_source_reference_v isr\n"
            "    ON isr.patient_id = p.patient_id AND isr.empi_rank = 1\n"
            "  WHERE p.last_date_of_visit >= :since\n"
            "  GROUP BY p.practice_name\n"
            "  ORDER BY patient_count DESC;"
        ),
    },
    {
        "view": "",
        "kind": "examples",
        "text": (
            "Q: For one source_id, the most recent A1C result (LOINC 4548-4).\n"
            "SQL:\n"
            "  SELECT result, clean_result, unit, datetime\n"
            "  FROM {p}federated_results_v\n"
            "  WHERE source_id = :source_id\n"
            "    AND code = '4548-4' AND code_type = 'LOINC'\n"
            "  ORDER BY datetime DESC\n"
            "  LIMIT 1;"
        ),
    },
]


def all_seed_docs() -> List[_Doc]:
    return [*SCHEMA_DOCS, *IDENTITY_DOCS, *FRESHNESS_DOCS, *EXAMPLES_DOCS, *RUN_SQL_EXAMPLES]
