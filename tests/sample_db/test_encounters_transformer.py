# tests/sample_db/test_encounters_transformer.py
from scripts.sample_db.transformers.encounters import transform_encounters


def test_encounters_produces_one_row_per_input(encounters_csv, ctx):
    # source map: Synthea patient ID → source_id, built earlier in real ETL.
    src_map = {f"pat-{i:03d}": f"src-pat-{i:03d}" for i in range(1, 4)}
    transform_encounters(encounters_csv, src_map, ctx)
    rows = ctx.output["dw.federated_encounters_v"]
    assert len(rows) == 3


def test_encounter_columns_match(encounters_csv, ctx):
    src_map = {f"pat-{i:03d}": f"src-pat-{i:03d}" for i in range(1, 4)}
    transform_encounters(encounters_csv, src_map, ctx)
    rows = ctx.output["dw.federated_encounters_v"]
    expected = {
        "source_id",
        "encounter_id",
        "type",
        "status",
        "status_datetime",
        "datetime",
        "location",
        "rendering_provider",
        "facility_name",
        "place_of_service",
    }
    assert expected.issubset(rows[0].keys())


def test_encounter_type_mapped_to_warehouse_vocab(encounters_csv, ctx):
    src_map = {f"pat-{i:03d}": f"src-pat-{i:03d}" for i in range(1, 4)}
    transform_encounters(encounters_csv, src_map, ctx)
    types = {r["type"] for r in ctx.output["dw.federated_encounters_v"]}
    # ambulatory → office_visit, inpatient → inpatient.
    assert "office_visit" in types
    assert "inpatient" in types


def test_unknown_patient_is_skipped(encounters_csv, ctx):
    # Only map one patient; others should be dropped.
    src_map = {"pat-001": "src-pat-001"}
    transform_encounters(encounters_csv, src_map, ctx)
    rows = ctx.output["dw.federated_encounters_v"]
    assert len(rows) == 1
    assert rows[0]["source_id"] == "src-pat-001"
