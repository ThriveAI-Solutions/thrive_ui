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
