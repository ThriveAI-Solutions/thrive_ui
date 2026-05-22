# tests/sample_db/test_meds_transformer.py
from scripts.sample_db.transformers.meds import transform_meds


def test_meds_one_row_per_medication(medications_csv, ctx):
    src_map = {f"pat-{i:03d}": f"src-pat-{i:03d}" for i in range(1, 4)}
    transform_meds(medications_csv, src_map, ctx)
    rows = ctx.output["dw.federated_meds_v"]
    assert len(rows) == 2


def test_meds_columns(medications_csv, ctx):
    src_map = {f"pat-{i:03d}": f"src-pat-{i:03d}" for i in range(1, 4)}
    transform_meds(medications_csv, src_map, ctx)
    expected = {
        "source_id",
        "ndc_code",
        "rxnorm_code",
        "med_name",
        "date_prescribed",
        "prescribing_provider_npi",
        "med_strength",
        "med_strength_unit",
        "med_form",
        "med_sig",
        "drug_supply_days",
        "number_of_refills",
    }
    rows = ctx.output["dw.federated_meds_v"]
    assert expected.issubset(rows[0].keys())


def test_known_rxnorm_gets_crosswalked_ndc(medications_csv, ctx):
    src_map = {f"pat-{i:03d}": f"src-pat-{i:03d}" for i in range(1, 4)}
    transform_meds(medications_csv, src_map, ctx)
    rows = ctx.output["dw.federated_meds_v"]
    by_rxnorm = {r["rxnorm_code"]: r for r in rows}
    assert by_rxnorm["6809"]["ndc_code"] == "00093-1054-01"


def test_unmapped_rxnorm_gets_placeholder_ndc(ctx):
    import pandas as pd

    rows_in = pd.DataFrame(
        [
            {
                "START": "2026-01-01T00:00:00Z",
                "STOP": "",
                "PATIENT": "pat-001",
                "PAYER": "p",
                "ENCOUNTER": "enc-001",
                "CODE": "99999",
                "DESCRIPTION": "Unknown med",
                "BASE_COST": 0,
                "PAYER_COVERAGE": 0,
                "DISPENSES": 1,
                "TOTALCOST": 0,
                "REASONCODE": "",
                "REASONDESCRIPTION": "",
            }
        ]
    )
    transform_meds(rows_in, {"pat-001": "src-pat-001"}, ctx)
    rows = ctx.output["dw.federated_meds_v"]
    assert rows[0]["ndc_code"] == "99999-9999-99"
    assert rows[0]["rxnorm_code"] == "99999"
