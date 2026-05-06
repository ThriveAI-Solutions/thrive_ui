-- Synthetic Redshift mirror for unit tests. Mirrors the §7.12 whitelist
-- views as SQLite tables. Keep this small (≤30 rows total).

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
    conditions TEXT
);
INSERT INTO internal_patient_profile_v VALUES
    (1, 'John', 'Smith', 'John Smith', '1962-05-01', 64, 'M', '2026-04-01', 'Buffalo Medical Group', 'Dr. Foo', 'diabetes'),
    (2, 'John', 'Smith', 'John Smith', '1971-08-12', 54, 'M', '2026-03-15', 'Kaleida Methodist', 'Dr. Bar', NULL),
    (3, 'Jane', 'Smith', 'Jane Smith', '1985-02-20', 41, 'F', '2026-04-20', 'ECMC', 'Dr. Baz', NULL);

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
    (3, 'src-jane-1985', 1, 'ECMC', 'EHR');

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
    ('src-john-1971', 'enc-3', 'inpatient', 'completed', '2026-03-15 14:00', '2026-03-15 14:00', 'Kaleida Methodist', 'Dr. Bar', 'Kaleida Methodist', '21');

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
    ('src-john-1962', '0DTJ4ZZ', 'ICD-10-PCS', 'Resection of appendix, percutaneous endoscopic', '2024-08-22', '2024-08-22', 'N', '1234567890');

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
    ('src-john-1962', '00093-7146-56', '18631', 'Azithromycin', '2026-04-02 14:00', '1234567890', '250', 'mg', 'Tab', 'Take 2 tabs day 1, 1 tab daily x4', 5, 0);

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
    ('src-john-1962', 'LOC-X-1', '', 'Local x-ray', '2024-08-20 09:00', '2024-08-20 09:00', '22', 'completed');

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

CREATE TABLE federated_claims_icd_procedure_detail_v (
    source_id TEXT,
    icd_procedure_code TEXT,
    code_type TEXT,
    procedure_description TEXT,
    procedure_date DATE,
    place_of_service TEXT,
    rendering_provider_npi TEXT,
    facility_name TEXT
);
INSERT INTO federated_claims_icd_procedure_detail_v VALUES
    ('src-john-1962', '0DTJ4ZZ', 'ICD-10-PCS', 'Resection of appendix, percutaneous endoscopic', '2024-08-22', '21', '1234567890', 'Buffalo General');

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

CREATE TABLE metric_federated_data_v (
    patient_id INTEGER,
    code TEXT,
    code_type TEXT,
    event_name TEXT,
    start_date DATE,
    is_claims_data INTEGER,
    data_source TEXT
);
INSERT INTO metric_federated_data_v VALUES
    (1, 'E11.9', 'ICD-10', 'Type 2 diabetes mellitus', '2024-06-12', 0, 'BMG');
