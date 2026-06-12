-- Synthetic Redshift mirror for unit tests. Mirrors the §7.12 whitelist
-- views as SQLite tables. Keep this small (≤30 rows total).

DROP TABLE IF EXISTS federated_adt_v;
DROP TABLE IF EXISTS internal_patient_profile_v;
DROP TABLE IF EXISTS internal_source_reference_v;
DROP TABLE IF EXISTS federated_demographic_v;
DROP TABLE IF EXISTS federated_encounters_v;
DROP TABLE IF EXISTS federated_problems_v;
DROP TABLE IF EXISTS federated_results_v;
DROP TABLE IF EXISTS federated_meds_v;
DROP TABLE IF EXISTS federated_orders_v;
DROP TABLE IF EXISTS federated_vaccination_v;
DROP TABLE IF EXISTS federated_vitals_v;
DROP TABLE IF EXISTS federated_documents_v;
DROP TABLE IF EXISTS federated_allergies_v;
DROP TABLE IF EXISTS federated_claims_icd_procedure_detail_v;
DROP TABLE IF EXISTS metric_federated_data_v;

-- Two patients named "John Smith" + one Jane Smith. Tests disambiguation.
CREATE TABLE internal_patient_profile_v (
    patient_id INTEGER PRIMARY KEY,
    first_name TEXT,
    last_name TEXT,
    full_name TEXT,
    date_of_birth DATE,
    age INTEGER,
    gender TEXT,
    last_date_of_visit DATE,
    practice_name TEXT,
    provider_name TEXT,
    conditions TEXT,
    zip_code TEXT,
    city TEXT,
    state TEXT
);
INSERT INTO internal_patient_profile_v VALUES
    (1, 'John', 'Smith', 'John Smith', '1962-05-01', 64, 'M', '2026-04-01', 'Buffalo Medical Group', 'Dr. Foo', 'diabetes', '14223', 'Buffalo', 'NY'),
    (2, 'John', 'Smith', 'John Smith', '1971-08-12', 54, 'M', '2026-03-15', 'Kaleida Methodist', 'Dr. Bar', NULL, '14201', 'Buffalo', 'NY'),
    (3, 'Jane', 'Smith', 'Jane Smith', '1985-02-20', 41, 'F', '2026-04-20', 'ECMC', 'Dr. Baz', NULL, '15213', 'Pittsburgh', 'PA'),
    (4, 'Mary', 'Jones', 'Mary Jones', '1956-03-10', 70, 'F', '2026-04-15', 'Kaleida Methodist', 'Dr. Foo', 'diabetes, hypertension', '14214', 'Buffalo', 'NY'),
    (5, 'Robert', 'Lee',  'Robert Lee',  '1970-11-22', 55, 'M', '2026-04-10', 'Buffalo Medical Group', 'Dr. Bar', 'hyperlipidemia', '14202', 'Buffalo', 'NY'),
    (6, 'Anne',   'Garcia','Anne Garcia', '1948-01-05', 78, 'F', '2024-08-01', 'Kaleida Methodist', 'Dr. Baz', 'hypertension', '14209', 'Buffalo', 'NY'),
    (7, 'Daniel', 'Wright','Daniel Wright','1977-09-30', 48, 'M', '2026-04-25', 'Buffalo Medical Group', 'Dr. Foo', 'type 2 diabetes mellitus', '14216', 'Buffalo', 'NY'),
    (8, 'Susan',  'Park',  'Susan Park',   '1955-07-14', 71, 'F', '2026-03-20', 'Kaleida Methodist', 'Dr. Bar', 'diabetes', '14204', 'Buffalo', 'NY');

CREATE TABLE internal_source_reference_v (
    patient_id INTEGER,
    source_id TEXT,
    empi_rank INTEGER,
    source_name TEXT,
    source_type TEXT
);
INSERT INTO internal_source_reference_v VALUES
    (1, 'src-john-1962', 1, 'BMG', 'EHR'),
    (1, 'src-john-1962-alt', 2, 'BMG', 'EHR'),
    (1, 'src-john-1962-stale', 99, 'BMG', 'EHR'),
    (2, 'src-john-1971', 1, 'Kaleida', 'EHR'),
    (3, 'src-jane-1985', 1, 'ECMC', 'EHR'),
    (4, 'src-mary-1956', 1, 'Kaleida', 'EHR'),
    (5, 'src-robert-1970', 1, 'BMG', 'EHR'),
    (6, 'src-anne-1948', 1, 'Kaleida', 'EHR'),
    (7, 'src-daniel-1977', 1, 'BMG', 'EHR'),
    (8, 'src-susan-1955', 1, 'Kaleida', 'EHR');

CREATE TABLE federated_encounters_v (
    source_id TEXT,
    encounter_id TEXT,
    type TEXT,
    status TEXT,
    status_datetime TIMESTAMP,
    datetime TIMESTAMP,
    location TEXT,
    rendering_provider TEXT,
    facility_name TEXT,
    place_of_service TEXT
);
INSERT INTO federated_encounters_v VALUES
    ('src-john-1962', 'enc-1', 'office_visit', 'completed', '2026-04-01 09:30', '2026-04-01 09:30', 'Buffalo Medical Group', 'Dr. Foo', 'Buffalo Medical Group', '11'),
    ('src-john-1962', 'enc-2', 'office_visit', 'completed', '2026-02-15 10:00', '2026-02-15 10:00', 'Buffalo Medical Group', 'Dr. Foo', 'Buffalo Medical Group', '11'),
    ('src-john-1971', 'enc-3', 'inpatient', 'completed', '2026-03-15 14:00', '2026-03-15 14:00', 'Kaleida Methodist', 'Dr. Bar', 'Kaleida Methodist', '21'),
    ('src-john-1962', 'enc-surg-1', 'inpatient', 'completed', '2025-06-15 08:00', '2025-06-15 08:00', 'Buffalo General', 'Dr. Ortho', 'Buffalo General', '21'),
    ('src-john-1962', 'enc-surg-2', 'outpatient', 'completed', '2025-06-15 14:00', '2025-06-15 14:00', 'Buffalo General', 'Dr. Anesthesia', 'Buffalo General', '21');

CREATE TABLE federated_demographic_v (
    source_id TEXT,
    patient_id TEXT,
    first_name TEXT,
    last_name TEXT,
    date_of_birth DATE,
    gender TEXT
);
INSERT INTO federated_demographic_v VALUES
    ('src-john-1962', '1', 'John', 'Smith', '1962-05-01', 'M'),
    ('src-john-1971', '2', 'John', 'Smith', '1971-08-12', 'M'),
    ('src-jane-1985', '3', 'Jane', 'Smith', '1985-02-20', 'F');

-- Phase 2 domains -----------------------------------------------------

CREATE TABLE federated_results_v (
    source_id TEXT,
    code TEXT,
    code_type TEXT,
    name TEXT,
    mnemonic TEXT,
    result TEXT,
    clean_result TEXT,
    unit TEXT,
    datetime TIMESTAMP,
    service_provider TEXT
);
INSERT INTO federated_results_v VALUES
    ('src-john-1962', '5195-3', 'LOINC', 'Hepatitis B sAg (HBsAg)', 'HBSAG', 'NEG', 'negative', NULL, '2026-02-10 08:00', 'BMG Lab'),
    ('src-john-1962', '4548-4', 'LOINC', 'Hemoglobin A1c', 'HBA1C', '7.2', '7.2', '%', '2026-03-15 09:00', 'BMG Lab'),
    ('src-john-1962', 'LOC-CHEM-99', 'local', 'Local Panel', 'LOCAL', 'POS', 'positive', NULL, '2026-01-05 11:00', 'BMG Lab'),
    ('src-john-1962', '22501-7', 'LOINC', 'Measles IgG Ab', 'MEASIGG', 'POS', 'positive', 'IU/mL', '2025-11-04 14:00', 'BMG Lab');

CREATE TABLE federated_problems_v (
    source_id TEXT,
    code TEXT,
    code_type TEXT,
    diagnosis TEXT,
    diagnosis_datetime TIMESTAMP,
    status_datetime TIMESTAMP,
    chronic_ind TEXT,
    service_provider_npi TEXT
);
INSERT INTO federated_problems_v VALUES
    ('src-john-1962', 'E11.9', 'ICD-10', 'Type 2 diabetes mellitus without complications', '2024-06-12', '2026-04-01', 'Y', '1234567890'),
    ('src-john-1962', 'B16.9', 'ICD-10', 'Acute hepatitis B without delta-agent', '2025-09-01', '2025-09-15', 'N', '1234567890'),
    ('src-john-1962', '0DTJ4ZZ', 'ICD-10-PCS', 'Resection of appendix, percutaneous endoscopic', '2024-08-22', '2024-08-22', 'N', '1234567890'),
    ('src-john-1962', '0WJG4ZZ', 'ICD-10-PCS', 'Inspection of peritoneal cavity, percutaneous endoscopic', '2025-01-10', '2025-01-10', 'N', '1234567890');

CREATE TABLE federated_meds_v (
    source_id TEXT,
    ndc_code TEXT,
    rxnorm_code TEXT,
    med_name TEXT,
    date_prescribed TIMESTAMP,
    prescribing_provider_npi TEXT,
    med_strength TEXT,
    med_strength_unit TEXT,
    med_form TEXT,
    med_sig TEXT,
    drug_supply_days INTEGER,
    number_of_refills INTEGER
);
INSERT INTO federated_meds_v VALUES
    ('src-john-1962', '00093-1054-01', '6809', 'Metformin', '2026-03-15 10:00', '1234567890', '500', 'mg', 'Tab', 'Take 1 tab BID', 90, 3),
    ('src-john-1962', '00093-7146-56', '18631', 'Azithromycin', '2026-04-02 14:00', '1234567890', '250', 'mg', 'Tab', 'Take 2 tabs day 1, 1 tab daily x4', 5, 0),
    -- src-mary-1956 has a Sulfa allergy (federated_allergies_v below) plus an
    -- active sulfamethoxazole prescription → drug-allergy conflict signal test.
    ('src-mary-1956', '49281-0790-15', '10180', 'Sulfamethoxazole', '2026-05-01 09:00', '0987654321', '800', 'mg', 'Tab', 'Take 1 tab BID x10', 10, 0);

CREATE TABLE federated_orders_v (
    source_id TEXT,
    code TEXT,
    code_type TEXT,
    name TEXT,
    date_of_procedure TIMESTAMP,
    order_created_date TIMESTAMP,
    place_of_service TEXT,
    status TEXT
);
INSERT INTO federated_orders_v VALUES
    ('src-john-1962', '71046', 'CPT', 'Chest X-ray, 2 views', '2026-03-15 13:00', '2026-03-15 09:00', '22', 'completed'),
    ('src-john-1962', '45378', 'CPT', 'Colonoscopy, diagnostic', '2025-12-01 10:00', '2025-11-20 09:00', '22', 'completed'),
    ('src-john-1962', 'LOC-X-1', '', 'Local x-ray', '2024-08-20 09:00', '2024-08-20 09:00', '22', 'completed'),
    ('src-john-1962', '27447', 'CPT', 'Total knee arthroplasty', '2025-06-15 08:00', '2025-06-10 09:00', '21', 'completed');

CREATE TABLE federated_vaccination_v (
    source_id TEXT,
    cvx TEXT,
    ndc TEXT,
    vaccine TEXT,
    manufacturer TEXT,
    lot TEXT,
    datetime TIMESTAMP,
    status_datetime TIMESTAMP,
    location_name TEXT,
    mvx_code TEXT
);
INSERT INTO federated_vaccination_v VALUES
    ('src-john-1962', '03', '00006-4681-00', 'Measles, Mumps, Rubella', 'Merck', 'LOT-MMR-01', '1968-04-15 09:00', '1968-04-15 09:00', 'BMG', 'MSD'),
    ('src-john-1962', '115', '49281-0400-15', 'Tdap', 'Sanofi', 'LOT-TDAP-22', '2020-09-10 11:00', '2020-09-10 11:00', 'BMG', 'PMC');

CREATE TABLE federated_documents_v (
    source_id TEXT,
    datetime TIMESTAMP,
    name TEXT,
    mnemonic TEXT,
    status TEXT,
    encounter_id TEXT,
    place_of_service TEXT,
    location_name TEXT
);
INSERT INTO federated_documents_v VALUES
    ('src-john-1962', '2026-04-01 09:30', 'Progress Note', 'PROGNOTE', 'final', 'enc-1', '11', 'Buffalo Medical Group'),
    ('src-john-1962', '2026-03-15 13:30', 'Radiology Report', 'XRREPORT', 'final', 'enc-r1', '22', 'Buffalo Medical Group'),
    ('src-john-1962', '2025-12-01 11:00', 'Procedure Note', 'PROCNOTE', 'final', 'enc-c1', '22', 'Buffalo Medical Group');

-- Schema corrected per redshift_tables.json (Task 4 reconciliation).
-- The real federated_claims_icd_procedure_detail_v has no source_id,
-- code_type, procedure_description, place_of_service, rendering_provider_npi,
-- or facility_name columns. The real columns are:
-- claim_line_identifier, icd_procedure_code, icd_type, primary_flag,
-- procedure_date, procedure_sequence_number, source_file_moyr,
-- source_file_name, source_format, source_name.
CREATE TABLE federated_claims_icd_procedure_detail_v (
    claim_line_identifier TEXT,
    icd_procedure_code TEXT,
    icd_type TEXT,
    primary_flag INTEGER,
    procedure_date DATE,
    procedure_sequence_number INTEGER,
    source_file_moyr DATE,
    source_file_name TEXT,
    source_format TEXT,
    source_name TEXT
);
INSERT INTO federated_claims_icd_procedure_detail_v VALUES
    ('CLM-001-01', '0DTJ4ZZ', 'ICD-10-PCS', 1, '2024-08-22', 1, '2024-08-01', 'claims_2024_08.csv', 'ICD', 'Highmark'),
    ('CLM-002-01', '0WJG4ZZ', 'ICD-10-PCS', 1, '2025-01-10', 1, '2025-01-01', 'claims_2025_01.csv', 'ICD', 'Highmark');

CREATE TABLE federated_vitals_v (
    source_id TEXT,
    code TEXT,
    code_type TEXT,
    name TEXT,
    result TEXT,
    clean_result TEXT,
    unit TEXT,
    datetime TIMESTAMP
);
INSERT INTO federated_vitals_v VALUES
    ('src-john-1962', '8480-6', 'LOINC', 'Systolic BP', '128', '128', 'mmHg', '2026-04-01 09:30');

-- Production federated_adt_v exposes patient_id only (no source_id, unlike
-- every other federated_*_v view). Identity is resolved by joining
-- internal_source_reference_v at empi_rank = 1.
CREATE TABLE federated_adt_v (
    patient_id INTEGER,
    event_date TIMESTAMP,
    event_location TEXT,
    location_type TEXT,
    clean_setting TEXT,
    status TEXT,
    admit_from TEXT,
    discharge_disposition TEXT,
    discharge_location TEXT
);
-- patient_id 1 maps to source_id 'src-john-1962' (empi_rank=1) per
-- internal_source_reference_v above. patient_id 2 maps to 'src-john-1971'.
INSERT INTO federated_adt_v VALUES
    (1, '2026-01-10 08:00', 'Buffalo General Hospital', 'Hospital', 'INPATIENT', 'Discharged', 'Home', 'Discharged to home', 'Home'),
    (1, '2025-06-15 07:30', 'Buffalo General Hospital', 'Hospital', 'INPATIENT', 'Discharged', 'Emergency Dept', 'Discharged to SNF', 'Sunrise SNF'),
    (1, '2025-06-20 10:00', 'Sunrise SNF', 'Skilled Nursing', 'SNF', 'Discharged', 'Buffalo General Hospital', 'Discharged to home', 'Home'),
    (1, '2024-11-05 14:00', 'ECMC Emergency', 'Emergency', 'EMERGENCY', 'Discharged', 'Self', 'Discharged to home', 'Home'),
    (2, '2026-03-15 14:00', 'Kaleida Methodist', 'Hospital', 'INPATIENT', 'Discharged', 'Home', 'Discharged to home', 'Home');

-- federated_allergies_v: dedicated allergies view per epic #201. Schema
-- inferred from the dev-warehouse evaluation set (allergy / type / severity
-- columns confirmed) plus the federated_problems_v shape. Verify columns
-- against production before shipping.
CREATE TABLE federated_allergies_v (
    source_id TEXT,
    code TEXT,
    code_type TEXT,
    allergy TEXT,
    type TEXT,
    severity TEXT,
    status TEXT,
    onset_date DATE,
    status_datetime TIMESTAMP,
    reaction TEXT,
    comments TEXT
);
-- src-john-1962: Penicillin (Drug, Severe, Active) + Peanuts (Food, Mild, Active)
-- + Latex (Adverse Reaction, Moderate, Resolved). Tests default-active filter,
-- include_inactive, category filter, text filter, snomed code filter, and
-- drug-allergy conflict signal (paired with amoxicillin med below).
-- src-john-1971: 'NO KNOWN ALLERGIES' negative assertion.
-- src-mary-1956: Sulfa drugs (Drug, Moderate, Active).
-- src-jane-1985: no rows → data_availability=no_records_found.
INSERT INTO federated_allergies_v VALUES
    ('src-john-1962', '91936005', 'SNOMED', 'Penicillin', 'Drug allergy', 'Severe', 'Active', '2010-05-01', '2010-05-01', 'Hives', NULL),
    ('src-john-1962', '91934008', 'SNOMED', 'Peanuts', 'Food allergy', 'Mild', 'Active', '2018-03-01', '2018-03-01', 'Rash', NULL),
    ('src-john-1962', '300916003', 'SNOMED', 'Latex', 'Adverse Reaction', 'Moderate', 'Resolved', '2015-01-01', '2020-06-01', 'Contact dermatitis', NULL),
    ('src-john-1971', NULL, NULL, 'NO KNOWN ALLERGIES', NULL, NULL, 'Active', NULL, '2024-01-15', NULL, NULL),
    ('src-mary-1956', '91937001', 'SNOMED', 'Sulfa drugs', 'Drug allergy', 'Moderate', 'Active', '2015-09-10', '2015-09-10', 'Rash', NULL);

CREATE TABLE metric_federated_data_v (
    patient_id INTEGER,
    origin_id TEXT,
    start_date DATE,
    end_date DATE,
    code TEXT,
    code_type TEXT,
    code_system TEXT,
    event_name TEXT,
    source_table TEXT,
    is_claims_data INTEGER,
    data_source TEXT
);
INSERT INTO metric_federated_data_v VALUES
    -- John 1962 — diabetes diagnosis (clinical)
    (1, 'enc-1', '2025-06-10', NULL, 'E11.9', 'ICD-10', 'ICD10CM', 'Type 2 diabetes mellitus', 'federated_problems_v', 0, 'BMG'),
    -- John 1971 — hypertension (not diabetic, useful negative)
    (2, 'enc-2', '2025-04-05', NULL, 'I10',   'ICD-10', 'ICD10CM', 'Essential hypertension',   'federated_problems_v', 0, 'Kaleida'),
    -- Mary Jones (70F Kaleida) — diabetes + metformin
    (4, 'enc-4',  '2025-09-12', NULL, 'E11.9', 'ICD-10', 'ICD10CM', 'Type 2 diabetes mellitus', 'federated_problems_v', 0, 'Kaleida'),
    (4, 'enc-4m', '2025-09-15', NULL, '6809',  'RxNorm', 'RXNORM',  'metformin',                 'federated_meds_v',     0, 'Kaleida'),
    -- Robert Lee — metformin without diagnosis
    (5, 'enc-5m', '2026-01-20', NULL, '6809',  'RxNorm', 'RXNORM',  'metformin',                 'federated_meds_v',     0, 'BMG'),
    -- Anne Garcia (78F Kaleida) — hypertension only, stale visit
    (6, 'enc-6',  '2024-07-22', NULL, 'I10',   'ICD-10', 'ICD10CM', 'Essential hypertension',   'federated_problems_v', 0, 'Kaleida'),
    -- Daniel Wright — diabetes + metformin via claims (refresh-cadence test)
    (7, 'enc-7c', '2025-11-30', NULL, 'E11.9', 'ICD-10', 'ICD10CM', 'Type 2 diabetes mellitus', 'federated_claims_icd_diagnosis_detail_v', 1, 'BMG'),
    (7, 'enc-7m', '2026-02-05', NULL, '6809',  'RxNorm', 'RXNORM',  'metformin',                 'federated_meds_v',     0, 'BMG'),
    -- Susan Park (71F Kaleida) — diabetes (qualifies "diabetics over 65 at Kaleida").
    -- Two E11.9 rows in the SAME month (2025-12) on purpose: a within-bucket
    -- multi-event fan-out. A month breakdown joins each diagnosis row, so Susan
    -- produces two rows in 2025-12 — COUNT(DISTINCT source_id) must still count
    -- her once. Asserted by test_diagnosis_month_breakdown_dedups_multi_event.
    (8, 'enc-8',  '2025-12-01', NULL, 'E11.9', 'ICD-10', 'ICD10CM', 'Type 2 diabetes mellitus', 'federated_problems_v', 0, 'Kaleida'),
    (8, 'enc-8b', '2025-12-20', NULL, 'E11.9', 'ICD-10', 'ICD10CM', 'Type 2 diabetes mellitus', 'federated_problems_v', 0, 'Kaleida');
