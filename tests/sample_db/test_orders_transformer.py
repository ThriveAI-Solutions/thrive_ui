from scripts.sample_db.transformers.orders import transform_orders


def test_orders_one_row_per_procedure(procedures_csv, ctx):
    src_map = {f"pat-{i:03d}": f"src-pat-{i:03d}" for i in range(1, 4)}
    transform_orders(procedures_csv, src_map, ctx)
    rows = ctx.output["dw.federated_orders_v"]
    assert len(rows) == 2


def test_orders_columns(procedures_csv, ctx):
    src_map = {f"pat-{i:03d}": f"src-pat-{i:03d}" for i in range(1, 4)}
    transform_orders(procedures_csv, src_map, ctx)
    rows = ctx.output["dw.federated_orders_v"]
    expected = {
        "source_id",
        "code",
        "code_type",
        "name",
        "date_of_procedure",
        "order_created_date",
        "place_of_service",
        "status",
    }
    assert expected.issubset(rows[0].keys())


def test_orders_empty_code_type_rate_at_scale(ctx):
    import pandas as pd

    rows_in = pd.DataFrame(
        [
            {
                "START": "2026-01-01T09:00:00Z",
                "STOP": "2026-01-01T10:00:00Z",
                "PATIENT": "pat-001",
                "ENCOUNTER": "enc-001",
                "CODE": "71046",
                "DESCRIPTION": "Chest X-ray",
                "BASE_COST": 100,
                "REASONCODE": "",
                "REASONDESCRIPTION": "",
            }
        ]
        * 1000
    )
    transform_orders(rows_in, {"pat-001": "src-pat-001"}, ctx)
    rows = ctx.output["dw.federated_orders_v"]
    empties = sum(1 for r in rows if r["code_type"] == "")
    # Spec target ~46%; allow ±7pp.
    assert 390 <= empties <= 530
