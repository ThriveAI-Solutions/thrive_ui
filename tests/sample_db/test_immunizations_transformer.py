# tests/sample_db/test_immunizations_transformer.py
from scripts.sample_db.transformers.immunizations import transform_immunizations


def test_immunizations_one_row_per_input(immunizations_csv, ctx):
    src_map = {f"pat-{i:03d}": f"src-pat-{i:03d}" for i in range(1, 4)}
    transform_immunizations(immunizations_csv, src_map, ctx)
    rows = ctx.output["dw.federated_vaccination_v"]
    assert len(rows) == 2


def test_immunization_columns(immunizations_csv, ctx):
    src_map = {f"pat-{i:03d}": f"src-pat-{i:03d}" for i in range(1, 4)}
    transform_immunizations(immunizations_csv, src_map, ctx)
    rows = ctx.output["dw.federated_vaccination_v"]
    expected = {
        "source_id",
        "cvx",
        "ndc",
        "vaccine",
        "manufacturer",
        "lot",
        "datetime",
        "status_datetime",
        "location_name",
        "mvx_code",
    }
    assert expected.issubset(rows[0].keys())


def test_cvx_populated_100pct(immunizations_csv, ctx):
    src_map = {f"pat-{i:03d}": f"src-pat-{i:03d}" for i in range(1, 4)}
    transform_immunizations(immunizations_csv, src_map, ctx)
    rows = ctx.output["dw.federated_vaccination_v"]
    assert all(r["cvx"] for r in rows)
    assert all(r["ndc"] for r in rows)
