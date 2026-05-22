# tests/sample_db/test_results_transformer.py
from scripts.sample_db.transformers.results import transform_results


def test_results_filters_to_lab_category(observations_csv, ctx):
    src_map = {f"pat-{i:03d}": f"src-pat-{i:03d}" for i in range(1, 4)}
    transform_results(observations_csv, src_map, ctx)
    rows = ctx.output.get("dw.federated_results_v", [])
    # Only one row in fixture has CATEGORY=laboratory.
    assert len(rows) == 1
    assert rows[0]["code"] == "4548-4"


def test_results_columns(observations_csv, ctx):
    src_map = {f"pat-{i:03d}": f"src-pat-{i:03d}" for i in range(1, 4)}
    transform_results(observations_csv, src_map, ctx)
    rows = ctx.output["dw.federated_results_v"]
    expected = {
        "source_id",
        "code",
        "code_type",
        "name",
        "mnemonic",
        "result",
        "clean_result",
        "unit",
        "datetime",
        "service_provider",
    }
    assert expected.issubset(rows[0].keys())


def test_results_code_type_empty_rate_at_scale(ctx):
    """At ~1000 rows, ~22% of code_types should be empty per spec §6.3."""
    import pandas as pd

    rows_in = pd.DataFrame(
        [
            {
                "DATE": "2026-03-15T09:00:00Z",
                "PATIENT": "pat-001",
                "ENCOUNTER": "enc-001",
                "CATEGORY": "laboratory",
                "CODE": "4548-4",
                "DESCRIPTION": "Hemoglobin A1c",
                "VALUE": "7.2",
                "UNITS": "%",
                "TYPE": "numeric",
            }
        ]
        * 1000
    )
    src_map = {"pat-001": "src-pat-001"}
    transform_results(rows_in, src_map, ctx)
    rows = ctx.output["dw.federated_results_v"]
    empty_count = sum(1 for r in rows if r["code_type"] == "")
    # Spec target ~22% empty; allow ±7pp.
    assert 150 <= empty_count <= 290
