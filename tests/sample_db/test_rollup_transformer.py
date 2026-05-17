# tests/sample_db/test_rollup_transformer.py
from scripts.sample_db.transformers.rollup import transform_rollup


def test_rollup_aggregates_problems_and_meds(ctx):
    # Seed ctx with upstream rows.
    ctx.add_rows(
        "dw.federated_problems_v",
        [
            {
                "source_id": "src-pat-001",
                "code": "E11.9",
                "code_type": "ICD-10",
                "diagnosis": "Type 2 diabetes",
                "diagnosis_datetime": "2025-06-10",
            },
        ],
    )
    ctx.add_rows(
        "dw.federated_meds_v",
        [
            {
                "source_id": "src-pat-001",
                "rxnorm_code": "6809",
                "ndc_code": "00093-1054-01",
                "med_name": "Metformin",
                "date_prescribed": "2025-09-15",
            },
        ],
    )
    src_to_patient = {"src-pat-001": 1}
    src_to_practice = {"src-pat-001": "BMG"}
    transform_rollup(src_to_patient, src_to_practice, ctx)
    rows = ctx.output["dw.metric_federated_data_v"]
    # 1 problem + 1 med.
    assert len(rows) == 2
    cols = {
        "patient_id",
        "origin_id",
        "start_date",
        "end_date",
        "code",
        "code_type",
        "code_system",
        "event_name",
        "source_table",
        "is_claims_data",
        "data_source",
    }
    assert cols.issubset(rows[0].keys())


def test_rollup_marks_claims_with_flag(ctx):
    ctx.add_rows(
        "dw.federated_claims_icd_diagnosis_detail_v",
        [
            {
                "claim_line_identifier": "CLM-1-L001",
                "icd_diagnosis_code": "E11.9",
                "icd_type": "ICD-10",
                "diagnosis_date": "2025-11-30",
            },
        ],
    )
    src_to_patient = {"src-pat-001": 1}
    src_to_practice = {"src-pat-001": "BMG"}
    # Need to bridge claim_line_identifier → source_id. Plan: rollup looks
    # up via a separately-built claim_to_source map.
    claim_to_source = {"CLM-1-L001": "src-pat-001"}
    transform_rollup(src_to_patient, src_to_practice, ctx, claim_to_source=claim_to_source)
    rows = ctx.output["dw.metric_federated_data_v"]
    claims_rows = [r for r in rows if r["is_claims_data"] == 1]
    assert len(claims_rows) == 1
