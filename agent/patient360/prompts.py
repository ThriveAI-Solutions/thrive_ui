"""Domain-specific summarizer prompts for Patient 360 (#246).

GENERATOR_VERSION stamps every summary; bump it on any change here so any
persisted/cached summaries invalidate.

SECTION_ORDER is thrive's clinical domains in a fixed review order. It mirrors
Chiron's order minus ``vitals`` (thrive's get_patient_clinical_data has no
vitals domain) and includes ``visits`` = encounters ∪ ADT.
"""

from __future__ import annotations

GENERATOR_VERSION = 1

SECTION_INPUT_TOKEN_BUDGET = 8000  # markdown table budget per section
SECTION_MAX_TOKENS = 700  # output cap per section narrative
MASTER_MAX_TOKENS = 1400

SECTION_ORDER = [
    "demographics",
    "encounters",
    "visits",
    "diagnoses",
    "medications",
    "labs",
    "procedures",
    "surgeries",
    "imaging",
    "immunizations",
    "admissions",
    "allergies",
]

HEDGING_RULES = (
    "\nRULES: State only what the data shows — never invent values, dosages, or dates. "
    "If told the table is a sample of N total rows, describe patterns and say the review "
    "covered a sample; never claim completeness. Quote cross-feed visit totals as "
    '"at least N". Write 1-3 tight paragraphs of clinical prose, no headings, no tables.'
)

_P = {
    "demographics": (
        "You are a clinical data analyst. Summarize this patient's demographics and "
        "attribution: age (from birth date), sex, practice attribution, source-record "
        "coverage, and recency of activity."
    ),
    "encounters": (
        "You are a clinical data analyst. Summarize this patient's encounter history: "
        "volume over time, care settings and facility types, recent encounters, and any "
        "notable clustering."
    ),
    "visits": (
        "You are a clinical data analyst. Summarize this patient's visit history from a "
        "UNION of the encounter and ADT feeds (a stay can appear in both feeds): inpatient "
        "vs outpatient vs ED mix, recent utilization, and any admission clusters."
    ),
    "diagnoses": (
        "You are a clinical data analyst. Summarize this patient's diagnoses: recurring and "
        "chronic conditions, major categories, and recency. Use the status column "
        "(active/inactive/resolved) when present; rows without a status are unknown — do not "
        "guess active-vs-resolved for those."
    ),
    "medications": (
        "You are a clinical data analyst. Summarize this patient's medications: recent and "
        "long-running drugs, drug classes, dosing patterns where shown (sig/quantity/days "
        "supply), and refill patterns. Never state a dosage not shown in the data."
    ),
    "labs": (
        "You are a clinical data analyst. Summarize this patient's laboratory results: "
        "notable abnormal values, trends over time in repeated tests, and the most recent "
        "values for key analytes."
    ),
    "procedures": (
        "You are a clinical data analyst. Summarize this patient's procedures: major "
        "procedures with dates, recurring procedure types, and recency."
    ),
    "surgeries": (
        "You are a clinical data analyst. Summarize this patient's surgical history: each "
        "surgery with its date and setting where shown."
    ),
    "imaging": (
        "You are a clinical data analyst. Summarize this patient's imaging history: "
        "modalities, body regions, dates, and repeated studies."
    ),
    "immunizations": (
        "You are a clinical data analyst. Summarize this patient's immunizations: vaccine "
        "types and dates, series where visible, and recency."
    ),
    "admissions": (
        "You are a clinical data analyst. Summarize this patient's ADT admissions: "
        "admission/discharge dates, facilities, and patterns of inpatient utilization."
    ),
    "allergies": (
        "You are a clinical data analyst. Summarize this patient's allergies: allergens, "
        "reactions and severities where recorded."
    ),
}

SECTION_PROMPTS = {k: v + HEDGING_RULES for k, v in _P.items()}

SYNTHESIS_SYSTEM_PROMPT = (
    "You are a clinical data analyst writing a chart-review summary for a healthcare "
    "analyst. You will receive one short narrative per clinical domain for a single "
    "patient. Synthesize them into a master summary in Markdown with EXACTLY these "
    "sections, in order: ## Overview, ## Problems & diagnoses, ## Medications, "
    "## Recent utilization, ## Notable results, ## Data coverage & caveats.\n"
    "Overview: 2-4 sentences — who the patient is and the shape of their record. "
    "Recent utilization: encounters/visits/admissions in roughly the last two years. "
    "Notable results: significant labs, imaging. "
    "Data coverage & caveats: which domains have no data, which section summaries were "
    "unavailable or based on samples, and the feed's limits (missing diagnosis status is "
    "unknown; cross-feed visit counts are minimums). Do not invent anything not present in "
    "the section narratives; do not add sections."
)
