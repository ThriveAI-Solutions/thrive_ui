-- Table: public.wny_health

DROP TABLE IF EXISTS public.wny_health;

CREATE TABLE IF NOT EXISTS public.wny_health
(
    research_id INTEGER NOT NULL,
    age INTEGER,
    sex TEXT,
    race_level_1 TEXT,
    race_level_2 TEXT,
    ethnicity TEXT,
    zip_code TEXT,
    county TEXT,
    adi_state TEXT,
    adi_national TEXT,  -- Nullable
    year INTEGER,
    asthma TEXT,
    diabetes TEXT,
    diabetes_poor_control TEXT,  -- Nullable
    diabetes_type TEXT,          -- Nullable
    hba1c_result REAL,           -- Nullable float values
    hypertension TEXT,
    bp_control TEXT,             -- Nullable
    bp_result TEXT,              -- Nullable
    obesity TEXT,
    prediabetes TEXT,
    pre_diabetes_diagnosis_type TEXT,  -- Nullable
    tobacco TEXT,
    breast_cancer_screening TEXT,
    cervical_cancer_screening TEXT,
    colorectal_cancer_screening TEXT
)

TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.wny_health
    OWNER to postgres;


COPY wny_health ("research_id","age","sex","race_level_1","race_level_2","ethnicity","zip_code","county","adi_state","adi_national","year","asthma","diabetes","diabetes_poor_control","diabetes_type","hba1c_result","hypertension","bp_control","bp_result","obesity","prediabetes","pre_diabetes_diagnosis_type","tobacco","breast_cancer_screening","cervical_cancer_screening","colorectal_cancer_screening")
FROM '/data/wny_health.csv'
DELIMITER ','
CSV HEADER;

select * from public.wny_health;