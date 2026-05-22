# tests/sample_db/test_claims_transformer.py
from scripts.sample_db.transformers.claims import transform_claims


def test_claims_produces_four_tables(claims_csv, encounters_csv, procedures_csv, ctx):
    src_map = {f"pat-{i:03d}": f"src-pat-{i:03d}" for i in range(1, 4)}
    transform_claims(claims_csv, encounters_csv, procedures_csv, src_map, ctx)
    for table in [
        "dw.federated_claims_icd_diagnosis_detail_v",
        "dw.federated_claims_icd_procedure_detail_v",
        "dw.federated_claims_medical_facility_detail_v",
        "dw.federated_claims_summary_v",
    ]:
        assert table in ctx.output, table


def test_diagnosis_detail_columns(claims_csv, encounters_csv, procedures_csv, ctx):
    src_map = {f"pat-{i:03d}": f"src-pat-{i:03d}" for i in range(1, 4)}
    transform_claims(claims_csv, encounters_csv, procedures_csv, src_map, ctx)
    rows = ctx.output["dw.federated_claims_icd_diagnosis_detail_v"]
    assert len(rows) >= 1
    assert "icd_diagnosis_code" in rows[0]


def test_procedure_detail_uses_correct_columns(claims_csv, encounters_csv, procedures_csv, ctx):
    """Real schema: claim_line_identifier, icd_procedure_code, icd_type, etc."""
    src_map = {f"pat-{i:03d}": f"src-pat-{i:03d}" for i in range(1, 4)}
    transform_claims(claims_csv, encounters_csv, procedures_csv, src_map, ctx)
    rows = ctx.output["dw.federated_claims_icd_procedure_detail_v"]
    if rows:
        cols = {"claim_line_identifier", "icd_procedure_code", "icd_type", "primary_flag", "procedure_date"}
        assert cols.issubset(rows[0].keys())
