# tests/sample_db/conftest.py
"""Tiny Synthea-shaped CSV fixtures for transformer tests."""

from __future__ import annotations

from io import StringIO

import pandas as pd
import pytest

from scripts.sample_db.transformers.base import TransformContext


@pytest.fixture
def ctx() -> TransformContext:
    return TransformContext(seed=42)


@pytest.fixture
def patients_csv() -> pd.DataFrame:
    data = """Id,BIRTHDATE,DEATHDATE,SSN,DRIVERS,PASSPORT,PREFIX,FIRST,MIDDLE,LAST,SUFFIX,MAIDEN,MARITAL,RACE,ETHNICITY,GENDER,BIRTHPLACE,ADDRESS,CITY,STATE,COUNTY,ZIP,LAT,LON,HEALTHCARE_EXPENSES,HEALTHCARE_COVERAGE,INCOME
pat-001,1962-05-01,,999-12-3456,S99912345,,,John,A,Smith,,,M,white,nonhispanic,M,Buffalo NY,123 Elm,Buffalo,New York,Erie,14223,42.9,-78.9,5000,3000,50000
pat-002,1985-02-20,,999-12-3457,S99912346,,,Jane,B,Smith,,,S,black,nonhispanic,F,Pittsburgh PA,456 Oak,Pittsburgh,Pennsylvania,Allegheny,15213,40.4,-79.9,3000,2000,40000
pat-003,1956-03-10,,999-12-3458,S99912347,,,Mary,C,Jones,,,M,white,nonhispanic,F,Buffalo NY,789 Pine,Buffalo,New York,Erie,14214,42.9,-78.9,4000,2500,45000
"""
    return pd.read_csv(StringIO(data))


@pytest.fixture
def encounters_csv() -> pd.DataFrame:
    data = """Id,START,STOP,PATIENT,ORGANIZATION,PROVIDER,PAYER,ENCOUNTERCLASS,CODE,DESCRIPTION,BASE_ENCOUNTER_COST,TOTAL_CLAIM_COST,PAYER_COVERAGE,REASONCODE,REASONDESCRIPTION
enc-001,2026-04-01T09:30:00Z,2026-04-01T10:00:00Z,pat-001,org-1,prov-1,payer-1,ambulatory,308335008,Patient encounter procedure,85.55,140.00,100.00,,
enc-002,2026-03-15T14:00:00Z,2026-03-15T16:00:00Z,pat-002,org-2,prov-2,payer-2,inpatient,183452005,Emergency hospital admission,500.00,2500.00,2000.00,44054006,Diabetes mellitus type 2
enc-003,2025-09-12T10:00:00Z,2025-09-12T10:30:00Z,pat-003,org-1,prov-1,payer-1,ambulatory,308335008,Patient encounter procedure,85.55,140.00,100.00,38341003,Hypertension
"""
    return pd.read_csv(StringIO(data))


@pytest.fixture
def conditions_csv() -> pd.DataFrame:
    data = """START,STOP,PATIENT,ENCOUNTER,CODE,DESCRIPTION
2024-06-12,,pat-001,enc-001,44054006,Type 2 diabetes mellitus
2025-09-12,,pat-003,enc-003,38341003,Essential hypertension
2024-08-01,,pat-001,enc-001,55822004,Hyperlipidemia
"""
    return pd.read_csv(StringIO(data))


@pytest.fixture
def medications_csv() -> pd.DataFrame:
    data = """START,STOP,PATIENT,PAYER,ENCOUNTER,CODE,DESCRIPTION,BASE_COST,PAYER_COVERAGE,DISPENSES,TOTALCOST,REASONCODE,REASONDESCRIPTION
2026-03-15T10:00:00Z,,pat-001,payer-1,enc-001,6809,Metformin 500 MG,10.00,8.00,3,30.00,44054006,Type 2 diabetes
2026-03-15T11:00:00Z,,pat-003,payer-1,enc-003,617318,Lisinopril 10 MG,5.00,4.00,3,15.00,38341003,Hypertension
"""
    return pd.read_csv(StringIO(data))


@pytest.fixture
def observations_csv() -> pd.DataFrame:
    data = """DATE,PATIENT,ENCOUNTER,CATEGORY,CODE,DESCRIPTION,VALUE,UNITS,TYPE
2026-03-15T09:00:00Z,pat-001,enc-001,laboratory,4548-4,Hemoglobin A1c/Hemoglobin.total in Blood,7.2,%,numeric
2026-04-01T09:30:00Z,pat-001,enc-001,vital-signs,8480-6,Systolic Blood Pressure,128,mmHg,numeric
2026-04-01T09:30:00Z,pat-001,enc-001,vital-signs,8462-4,Diastolic Blood Pressure,82,mmHg,numeric
2025-09-12T10:00:00Z,pat-003,enc-003,vital-signs,8480-6,Systolic Blood Pressure,150,mmHg,numeric
"""
    return pd.read_csv(StringIO(data))


@pytest.fixture
def procedures_csv() -> pd.DataFrame:
    data = """START,STOP,PATIENT,ENCOUNTER,CODE,DESCRIPTION,BASE_COST,REASONCODE,REASONDESCRIPTION
2026-03-15T13:00:00Z,2026-03-15T13:30:00Z,pat-001,enc-001,71046,Chest X-ray 2 views,100.00,,
2025-12-01T10:00:00Z,2025-12-01T11:00:00Z,pat-001,enc-001,45378,Colonoscopy diagnostic,800.00,,
"""
    return pd.read_csv(StringIO(data))


@pytest.fixture
def immunizations_csv() -> pd.DataFrame:
    data = """DATE,PATIENT,ENCOUNTER,CODE,DESCRIPTION,BASE_COST
2020-09-10T11:00:00Z,pat-001,enc-001,115,Tdap,40.00
2026-01-15T09:00:00Z,pat-003,enc-003,140,Influenza seasonal injectable,25.00
"""
    return pd.read_csv(StringIO(data))


@pytest.fixture
def providers_csv() -> pd.DataFrame:
    data = """Id,ORGANIZATION,NAME,GENDER,SPECIALITY,ADDRESS,CITY,STATE,ZIP,LAT,LON,ENCOUNTERS,PROCEDURES
prov-1,org-1,Dr. Foo Bar,M,GENERAL PRACTICE,123 Med,Buffalo,New York,14223,42.9,-78.9,100,50
prov-2,org-2,Dr. Baz Qux,F,EMERGENCY MEDICINE,456 Hosp,Buffalo,New York,14201,42.9,-78.9,200,100
"""
    return pd.read_csv(StringIO(data))


@pytest.fixture
def organizations_csv() -> pd.DataFrame:
    data = """Id,NAME,ADDRESS,CITY,STATE,ZIP,LAT,LON,PHONE,REVENUE,UTILIZATION
org-1,Buffalo Medical Group,123 Med,Buffalo,New York,14223,42.9,-78.9,716-555-0100,5000000,500
org-2,Kaleida Methodist,456 Hosp,Buffalo,New York,14201,42.9,-78.9,716-555-0200,20000000,2000
"""
    return pd.read_csv(StringIO(data))


@pytest.fixture
def claims_csv() -> pd.DataFrame:
    data = """Id,PATIENTID,PROVIDERID,PRIMARYPATIENTINSURANCEID,SECONDARYPATIENTINSURANCEID,DEPARTMENTID,PATIENTDEPARTMENTID,DIAGNOSIS1,DIAGNOSIS2,DIAGNOSIS3,DIAGNOSIS4,DIAGNOSIS5,DIAGNOSIS6,DIAGNOSIS7,DIAGNOSIS8,REFERRINGPROVIDERID,APPOINTMENTID,CURRENTILLNESSDATE,SERVICEDATE,SUPERVISINGPROVIDERID,STATUS1,STATUS2,STATUSP,OUTSTANDING1,OUTSTANDING2,OUTSTANDINGP,LASTBILLEDDATE1,LASTBILLEDDATE2,LASTBILLEDDATEP,HEALTHCARECLAIMTYPEID1,HEALTHCARECLAIMTYPEID2
clm-001,pat-001,prov-1,ins-1,,1,1,44054006,,,,,,,,prov-1,enc-001,2024-06-01T00:00:00Z,2026-04-01T09:30:00Z,prov-1,BILLED,,BILLED,0,0,0,2026-04-15T00:00:00Z,,2026-04-15T00:00:00Z,1,
clm-002,pat-003,prov-1,ins-1,,1,1,38341003,,,,,,,,prov-1,enc-003,2025-08-01T00:00:00Z,2025-09-12T10:00:00Z,prov-1,BILLED,,BILLED,0,0,0,2025-09-30T00:00:00Z,,2025-09-30T00:00:00Z,1,
"""
    return pd.read_csv(StringIO(data))
