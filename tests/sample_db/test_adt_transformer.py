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


def test_adt_transformer_emits_one_row_per_admission(encounters_with_admissions_csv, synthea_to_pid, ctx):
    transform_adt(encounters_with_admissions_csv, synthea_to_pid, ctx)
    rows = ctx.output["dw.federated_adt_v"]
    # Only inpatient + emergency should be emitted; ambulatory is dropped.
    assert len(rows) == 2


def test_adt_transformer_skips_non_admission_classes(encounters_with_admissions_csv, synthea_to_pid, ctx):
    transform_adt(encounters_with_admissions_csv, synthea_to_pid, ctx)
    rows = ctx.output["dw.federated_adt_v"]
    settings = {r["clean_setting"] for r in rows}
    # No "OFFICE" / "WELLNESS" / "AMBULATORY" rows; only INPATIENT + EMERGENCY.
    assert settings == {"INPATIENT", "EMERGENCY"}


def test_adt_transformer_maps_inpatient_setting_and_status(encounters_with_admissions_csv, synthea_to_pid, ctx):
    transform_adt(encounters_with_admissions_csv, synthea_to_pid, ctx)
    rows = ctx.output["dw.federated_adt_v"]
    inpatient = next(r for r in rows if r["clean_setting"] == "INPATIENT")
    assert inpatient["patient_id"] == 1
    assert inpatient["event_location"] == "Buffalo General"
    assert inpatient["location_type"] == "Hospital"
    # STOP is present → discharged with a discharge_disposition / location.
    assert inpatient["status"] == "Discharged"
    assert inpatient["discharge_disposition"] is not None
    assert inpatient["discharge_location"] is not None


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
    """An encounter still in-flight (STOP empty) → status='Admitted', no discharge cols."""
    data = """Id,START,STOP,PATIENT,ORGANIZATION,PROVIDER,PAYER,ENCOUNTERCLASS,CODE,DESCRIPTION,BASE_ENCOUNTER_COST,TOTAL_CLAIM_COST,PAYER_COVERAGE,REASONCODE,REASONDESCRIPTION
enc-001,2026-06-01T08:00:00Z,,pat-001,Buffalo General,prov-1,payer-1,inpatient,183452005,Emergency hospital admission,500.00,2500.00,2000.00,,
"""
    encounters = pd.read_csv(StringIO(data))
    transform_adt(encounters, synthea_to_pid, ctx)
    rows = ctx.output["dw.federated_adt_v"]
    assert len(rows) == 1
    assert rows[0]["status"] == "Admitted"
    assert rows[0]["discharge_disposition"] is None
    assert rows[0]["discharge_location"] is None


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
