# tests/sample_db/test_identity_transformer.py
from scripts.sample_db.transformers.identity import transform_identity


def test_identity_produces_four_tables(patients_csv, encounters_csv, ctx):
    transform_identity(patients_csv, encounters_csv, ctx)
    assert "dw.internal_patient_profile_v" in ctx.output
    assert "dw.internal_source_reference_v" in ctx.output
    assert "dw.federated_demographic_v" in ctx.output
    assert "dw.federated_demographic_history_v" in ctx.output


def test_patient_profile_row_per_patient(patients_csv, encounters_csv, ctx):
    transform_identity(patients_csv, encounters_csv, ctx)
    profiles = ctx.output["dw.internal_patient_profile_v"]
    assert len(profiles) == 3
    cols = {
        "patient_id",
        "first_name",
        "last_name",
        "full_name",
        "date_of_birth",
        "age",
        "gender",
        "last_date_of_visit",
        "practice_name",
        "provider_name",
        "conditions",
        "zip_code",
        "city",
        "state",
    }
    assert cols.issubset(profiles[0].keys())


def test_full_name_concatenated(patients_csv, encounters_csv, ctx):
    transform_identity(patients_csv, encounters_csv, ctx)
    profiles = ctx.output["dw.internal_patient_profile_v"]
    by_id = {p["patient_id"]: p for p in profiles}
    # patient IDs are sequential integers, not Synthea UUIDs.
    sample = next(iter(profiles))
    assert sample["full_name"] == f"{sample['first_name']} {sample['last_name']}"


def test_source_reference_has_at_least_one_active_per_patient(patients_csv, encounters_csv, ctx):
    transform_identity(patients_csv, encounters_csv, ctx)
    refs = ctx.output["dw.internal_source_reference_v"]
    active_by_patient = {}
    for r in refs:
        if r["empi_rank"] != 99:
            active_by_patient.setdefault(r["patient_id"], 0)
            active_by_patient[r["patient_id"]] += 1
    # Every patient has >=1 active source.
    assert len(active_by_patient) == 3


def test_demographic_v_row_per_source(patients_csv, encounters_csv, ctx):
    transform_identity(patients_csv, encounters_csv, ctx)
    demo = ctx.output["dw.federated_demographic_v"]
    refs = ctx.output["dw.internal_source_reference_v"]
    # one demographic row per active source.
    active_source_ids = {r["source_id"] for r in refs if r["empi_rank"] != 99}
    demo_source_ids = {d["source_id"] for d in demo}
    assert active_source_ids == demo_source_ids


def test_zip_code_propagated(patients_csv, encounters_csv, ctx):
    transform_identity(patients_csv, encounters_csv, ctx)
    profiles = ctx.output["dw.internal_patient_profile_v"]
    zips = {p["zip_code"] for p in profiles}
    assert "14223" in zips  # Buffalo from fixture
