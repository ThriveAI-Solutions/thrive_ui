# tests/sample_db/test_documents_transformer.py
from scripts.sample_db.transformers.documents import transform_documents


def test_documents_at_least_one_per_encounter(encounters_csv, ctx):
    src_map = {f"pat-{i:03d}": f"src-pat-{i:03d}" for i in range(1, 4)}
    transform_documents(encounters_csv, src_map, ctx)
    rows = ctx.output["dw.federated_documents_v"]
    assert len(rows) >= 3


def test_documents_columns(encounters_csv, ctx):
    src_map = {f"pat-{i:03d}": f"src-pat-{i:03d}" for i in range(1, 4)}
    transform_documents(encounters_csv, src_map, ctx)
    rows = ctx.output["dw.federated_documents_v"]
    expected = {
        "source_id",
        "datetime",
        "name",
        "mnemonic",
        "status",
        "encounter_id",
        "place_of_service",
        "location_name",
    }
    assert expected.issubset(rows[0].keys())


def test_no_body_columns_present(encounters_csv, ctx):
    src_map = {f"pat-{i:03d}": f"src-pat-{i:03d}" for i in range(1, 4)}
    transform_documents(encounters_csv, src_map, ctx)
    rows = ctx.output["dw.federated_documents_v"]
    for r in rows:
        assert "body" not in r and "content" not in r and "narrative" not in r


def test_document_names_drawn_from_real_distribution(encounters_csv, ctx):
    import pandas as pd

    fixture = pd.concat([encounters_csv] * 50, ignore_index=True)
    src_map = {f"pat-{i:03d}": f"src-pat-{i:03d}" for i in range(1, 4)}
    transform_documents(fixture, src_map, ctx)
    names = {r["name"] for r in ctx.output["dw.federated_documents_v"]}
    # Should include at least the most common buckets.
    assert "Progress Note" in names or "Visit Note" in names
