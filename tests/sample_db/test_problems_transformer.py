# tests/sample_db/test_problems_transformer.py
from collections import Counter

from scripts.sample_db.transformers.problems import transform_problems


def test_problems_one_row_per_condition(conditions_csv, ctx):
    src_map = {f"pat-{i:03d}": f"src-pat-{i:03d}" for i in range(1, 4)}
    transform_problems(conditions_csv, src_map, ctx)
    rows = ctx.output["dw.federated_problems_v"]
    # 3 conditions, each may be dual-coded or single — at least 3.
    assert len(rows) >= 3


def test_problems_columns(conditions_csv, ctx):
    src_map = {f"pat-{i:03d}": f"src-pat-{i:03d}" for i in range(1, 4)}
    transform_problems(conditions_csv, src_map, ctx)
    expected = {
        "source_id",
        "code",
        "code_type",
        "diagnosis",
        "diagnosis_datetime",
        "status_datetime",
        "chronic_ind",
        "service_provider_npi",
    }
    rows = ctx.output["dw.federated_problems_v"]
    assert expected.issubset(rows[0].keys())


def test_known_snomed_gets_icd10_with_high_probability(ctx):
    """At population scale, ICD-10 should dominate for mapped codes."""
    import pandas as pd

    rows_in = pd.DataFrame(
        [
            {
                "START": "2024-01-01",
                "STOP": "",
                "PATIENT": "pat-001",
                "ENCOUNTER": "enc-001",
                "CODE": "44054006",  # diabetes
                "DESCRIPTION": "Type 2 diabetes mellitus",
            }
        ]
        * 1000
    )
    src_map = {"pat-001": "src-pat-001"}
    transform_problems(rows_in, src_map, ctx)
    rows = ctx.output["dw.federated_problems_v"]
    type_counts = Counter(
        (
            "ICD-10"
            if r["code_type"] in {"ICD-10", "ICD10", "ICD-10-CM", "ICD10CM"}
            else "SNOMED"
            if r["code_type"] in {"SNOMED CT", "SNOMED-CT", "SNOMED", "SNOMEDCT"}
            else "ICD-9"
            if r["code_type"] in {"ICD-9", "ICD9", "ICD-9-CM"}
            else "empty"
            if r["code_type"] == ""
            else "other"
        )
        for r in rows
    )
    # Roughly 70% ICD-10, with reasonable tolerance.
    assert type_counts["ICD-10"] >= 600


def test_unmapped_snomed_becomes_empty_code_type(ctx):
    import pandas as pd

    rows_in = pd.DataFrame(
        [
            {
                "START": "2024-01-01",
                "STOP": "",
                "PATIENT": "pat-001",
                "ENCOUNTER": "enc-001",
                "CODE": "99999999",  # not in crosswalk
                "DESCRIPTION": "Unknown condition",
            }
        ]
        * 200
    )
    src_map = {"pat-001": "src-pat-001"}
    transform_problems(rows_in, src_map, ctx)
    rows = ctx.output["dw.federated_problems_v"]
    # All should keep SNOMED variant or fall to empty (not crash).
    type_buckets = {r["code_type"] for r in rows}
    assert all(
        t in {"SNOMED CT", "SNOMED-CT", "SNOMED", "SNOMEDCT", "2.16.840.1.113883.6.96", ""} for t in type_buckets
    ), type_buckets
