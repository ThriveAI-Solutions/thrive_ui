# Crosswalks

Seed data mapping Synthea's native code systems to systems present in the
production Redshift warehouse:

- `snomed_to_icd10.csv` — Synthea condition SNOMED codes → ICD-10-CM.
- `rxnorm_to_ndc.csv` — Synthea medication RxNorm codes → most-common NDC.

## Format

CSV with header. First column is the source code (varchar), second is the
target code (varchar), third is a human-readable description for review.

## Provenance

These are hand-curated subsets, not full UMLS exports:

- ICD-10 mappings derived from CMS GEMs (General Equivalence Mappings),
  which are public domain. Source:
  https://www.cms.gov/medicare/coding-billing/icd-10-codes
- NDC values are representative picks for each RxNorm ingredient, sourced
  from the NLM RxNorm REST API (free, no auth):
  https://lhncbc.nlm.nih.gov/RxNav/APIs/RxNormAPIs.html

The seeds cover the most common Synthea outputs at population=1,000.
Unmapped codes are recorded by the ETL at regeneration time (see the
`--report-unmapped` flag on `etl.py`) and can be appended here.

## Regenerating from scratch

Not automated — manual curation. To extend coverage:

1. Run a sample ETL with `--report-unmapped` after a Synthea regeneration.
2. For each reported unmapped SNOMED, look it up in the CMS GEMs file.
3. For each reported unmapped RxNorm, query
   `https://rxnav.nlm.nih.gov/REST/rxcui/<rxcui>/ndcs.json` and pick the
   first NDC.
4. Append to the relevant CSV, re-run the ETL.
