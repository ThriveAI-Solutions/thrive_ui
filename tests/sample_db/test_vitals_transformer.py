# tests/sample_db/test_vitals_transformer.py
from scripts.sample_db.transformers.vitals import transform_vitals


def test_vitals_filters_to_vital_signs(observations_csv, ctx):
    src_map = {f"pat-{i:03d}": f"src-pat-{i:03d}" for i in range(1, 4)}
    transform_vitals(observations_csv, src_map, ctx)
    rows = ctx.output["dw.federated_vitals_v"]
    # 3 vital-signs in fixture.
    assert len(rows) == 3


def test_vitals_columns(observations_csv, ctx):
    src_map = {f"pat-{i:03d}": f"src-pat-{i:03d}" for i in range(1, 4)}
    transform_vitals(observations_csv, src_map, ctx)
    rows = ctx.output["dw.federated_vitals_v"]
    expected = {"source_id", "code", "code_type", "name", "result", "clean_result", "unit", "datetime"}
    assert expected.issubset(rows[0].keys())


def test_vitals_loinc_dominant(ctx):
    import pandas as pd

    rows_in = pd.DataFrame(
        [
            {
                "DATE": "2026-04-01T09:30:00Z",
                "PATIENT": "pat-001",
                "ENCOUNTER": "enc-001",
                "CATEGORY": "vital-signs",
                "CODE": "8480-6",
                "DESCRIPTION": "Systolic BP",
                "VALUE": "128",
                "UNITS": "mmHg",
                "TYPE": "numeric",
            }
        ]
        * 1000
    )
    transform_vitals(rows_in, {"pat-001": "src-pat-001"}, ctx)
    rows = ctx.output["dw.federated_vitals_v"]
    loinc_count = sum(1 for r in rows if r["code_type"] in {"LOINC", "LOINC-LN"})
    # Spec target ~91%; allow ±7pp.
    assert loinc_count >= 800
