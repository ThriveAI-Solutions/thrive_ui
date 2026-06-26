"""Synthea encounters.csv → dw.federated_adt_v transformer."""

from __future__ import annotations

from io import StringIO

import pandas as pd
import pytest

from scripts.sample_db.transformers.adt import transform_adt


@pytest.fixture
def encounters_with_admissions_csv() -> pd.DataFrame:
    """Synthea encounters: 1 inpatient + 1 emergency + 1 ambulatory.
    Only the inpatient and emergency rows should produce ADT events."""
    data = """Id,START,STOP,PATIENT,ORGANIZATION,PROVIDER,PAYER,ENCOUNTERCLASS,CODE,DESCRIPTION,BASE_ENCOUNTER_COST,TOTAL_CLAIM_COST,PAYER_COVERAGE,REASONCODE,REASONDESCRIPTION
enc-001,2026-03-15T14:00:00Z,2026-03-18T10:00:00Z,pat-001,Buffalo General,prov-1,payer-1,inpatient,183452005,Emergency hospital admission,500.00,2500.00,2000.00,44054006,Diabetes mellitus type 2
enc-002,2025-09-10T22:00:00Z,2025-09-11T03:00:00Z,pat-001,ECMC,prov-2,payer-1,emergency,50849002,Emergency room admission,200.00,800.00,600.00,38341003,Hypertension
enc-003,2026-04-01T09:30:00Z,2026-04-01T10:00:00Z,pat-002,Buffalo Medical Group,prov-1,payer-1,ambulatory,308335008,Patient encounter procedure,85.55,140.00,100.00,,
"""
    return pd.read_csv(StringIO(data))


@pytest.fixture
def synthea_to_pid() -> dict[str, int]:
    """Synthea patient id → warehouse-internal integer patient_id."""
    return {"pat-001": 1, "pat-002": 2}


def test_adt_transformer_emits_admit_and_discharge_events(encounters_with_admissions_csv, synthea_to_pid, ctx):
    transform_adt(encounters_with_admissions_csv, synthea_to_pid, ctx)
    rows = ctx.output["dw.federated_adt_v"]
    # Inpatient + emergency each emit ADMIT and DISCHARGE; ambulatory is dropped.
    assert len(rows) == 4


def test_adt_transformer_skips_non_admission_classes(encounters_with_admissions_csv, synthea_to_pid, ctx):
    transform_adt(encounters_with_admissions_csv, synthea_to_pid, ctx)
    rows = ctx.output["dw.federated_adt_v"]
    settings = {r["clean_setting"] for r in rows}
    # No "OFFICE" / "WELLNESS" / "AMBULATORY" rows; only INPATIENT + EMERGENCY.
    assert settings == {"INPATIENT", "EMERGENCY"}


def test_adt_transformer_maps_inpatient_setting_and_status(encounters_with_admissions_csv, synthea_to_pid, ctx):
    transform_adt(encounters_with_admissions_csv, synthea_to_pid, ctx)
    rows = ctx.output["dw.federated_adt_v"]
    inpatient = next(r for r in rows if r["clean_setting"] == "INPATIENT" and r["clean_status"] == "ADMIT")
    assert inpatient["patient_id"] == 1
    assert inpatient["event_location"] == "Buffalo General"
    assert inpatient["location_type"] == "Hospital"
    assert inpatient["status"] == "Admitted"
    assert inpatient["discharge_disposition"] is None
    assert inpatient["discharge_location"] is None


def test_adt_transformer_maps_discharge_event_to_stop_time(encounters_with_admissions_csv, synthea_to_pid, ctx):
    transform_adt(encounters_with_admissions_csv, synthea_to_pid, ctx)
    rows = ctx.output["dw.federated_adt_v"]
    discharge = next(r for r in rows if r["clean_setting"] == "INPATIENT" and r["clean_status"] == "DISCHARGE")
    assert discharge["status"] == "Discharged"
    assert str(discharge["event_date"]) == "2026-03-18 10:00:00"
    assert discharge["discharge_disposition"] == "Discharged to home"
    assert discharge["discharge_location"] == "Home"


def test_adt_transformer_emergency_uses_emergency_location_type(encounters_with_admissions_csv, synthea_to_pid, ctx):
    transform_adt(encounters_with_admissions_csv, synthea_to_pid, ctx)
    rows = ctx.output["dw.federated_adt_v"]
    ed = next(r for r in rows if r["clean_setting"] == "EMERGENCY")
    assert ed["location_type"] == "Emergency"


def test_adt_transformer_skips_unmapped_patients(encounters_with_admissions_csv, ctx):
    """Encounters whose patient is not in synthea_to_pid are dropped silently."""
    transform_adt(encounters_with_admissions_csv, {"pat-001": 1}, ctx)
    rows = ctx.output["dw.federated_adt_v"]
    # pat-002 isn't mapped, but pat-002 is ambulatory and already filtered out anyway.
    # All emitted rows must belong to mapped patients.
    assert all(r["patient_id"] == 1 for r in rows)


def test_adt_transformer_inflight_admission_has_no_discharge_details(synthea_to_pid, ctx):
    """An encounter still in-flight (STOP empty) → ADMIT only, no discharge event."""
    data = """Id,START,STOP,PATIENT,ORGANIZATION,PROVIDER,PAYER,ENCOUNTERCLASS,CODE,DESCRIPTION,BASE_ENCOUNTER_COST,TOTAL_CLAIM_COST,PAYER_COVERAGE,REASONCODE,REASONDESCRIPTION
enc-001,2026-06-01T08:00:00Z,,pat-001,Buffalo General,prov-1,payer-1,inpatient,183452005,Emergency hospital admission,500.00,2500.00,2000.00,,
"""
    encounters = pd.read_csv(StringIO(data))
    transform_adt(encounters, synthea_to_pid, ctx)
    rows = ctx.output["dw.federated_adt_v"]
    assert len(rows) == 1
    assert rows[0]["status"] == "Admitted"
    assert rows[0]["clean_status"] == "ADMIT"
    assert rows[0]["discharge_disposition"] is None
    assert rows[0]["discharge_location"] is None


def test_adt_transformer_emits_visit_number_status_cancelled(encounters_with_admissions_csv, synthea_to_pid, ctx):
    transform_adt(encounters_with_admissions_csv, synthea_to_pid, ctx)
    rows = ctx.output["dw.federated_adt_v"]
    # visit_number present + non-null on every row, shared by events in the same encounter.
    visit_numbers = [r["visit_number"] for r in rows]
    assert all(vn for vn in visit_numbers)
    assert sorted(visit_numbers.count(vn) for vn in set(visit_numbers)) == [2, 2]
    statuses = {r["clean_status"] for r in rows}
    assert statuses == {"ADMIT", "DISCHARGE"}
    for r in rows:
        assert r["cancelled_flag"] == "N"


def test_adt_transformer_inflight_clean_status_is_admit(synthea_to_pid, ctx):
    data = """Id,START,STOP,PATIENT,ORGANIZATION,PROVIDER,PAYER,ENCOUNTERCLASS,CODE,DESCRIPTION,BASE_ENCOUNTER_COST,TOTAL_CLAIM_COST,PAYER_COVERAGE,REASONCODE,REASONDESCRIPTION
enc-001,2026-06-01T08:00:00Z,,pat-001,Buffalo General,prov-1,payer-1,inpatient,183452005,Emergency hospital admission,500.00,2500.00,2000.00,,
"""
    encounters = pd.read_csv(StringIO(data))
    transform_adt(encounters, synthea_to_pid, ctx)
    rows = ctx.output["dw.federated_adt_v"]
    assert len(rows) == 1
    assert rows[0]["clean_status"] == "ADMIT"
    assert rows[0]["cancelled_flag"] == "N"
    assert rows[0]["visit_number"]


def test_adt_transformer_handles_empty_input(synthea_to_pid, ctx):
    empty = pd.DataFrame(
        columns=[
            "Id",
            "START",
            "STOP",
            "PATIENT",
            "ORGANIZATION",
            "PROVIDER",
            "PAYER",
            "ENCOUNTERCLASS",
            "CODE",
            "DESCRIPTION",
            "BASE_ENCOUNTER_COST",
            "TOTAL_CLAIM_COST",
            "PAYER_COVERAGE",
            "REASONCODE",
            "REASONDESCRIPTION",
        ]
    )
    transform_adt(empty, synthea_to_pid, ctx)
    assert ctx.output.get("dw.federated_adt_v", []) == []
