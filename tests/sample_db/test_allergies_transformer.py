"""Synthea allergies.csv → dw.federated_allergies_v transformer."""

from __future__ import annotations

from scripts.sample_db.transformers.allergies import transform_allergies


def test_allergies_transformer_emits_one_row_per_input(allergies_csv, ctx):
    source_map = {"pat-001": "src-001", "pat-003": "src-003"}
    transform_allergies(allergies_csv, source_map, ctx)
    rows = ctx.output["dw.federated_allergies_v"]
    assert len(rows) == 2


def test_allergies_transformer_skips_unmapped_patients(allergies_csv, ctx):
    """Rows for patients missing from source_map are dropped silently —
    matches the pattern in problems.py."""
    source_map = {"pat-001": "src-001"}  # pat-003 missing
    transform_allergies(allergies_csv, source_map, ctx)
    rows = ctx.output["dw.federated_allergies_v"]
    assert len(rows) == 1
    assert rows[0]["source_id"] == "src-001"


def test_allergies_transformer_maps_category_to_warehouse_type(allergies_csv, ctx):
    """Synthea CATEGORY 'medication' → warehouse 'Drug allergy'; 'food' →
    'Food allergy'. The eval set documents 'Drug allergy' / 'Food allergy'
    / 'Adverse Reaction' as the canonical values."""
    source_map = {"pat-001": "src-001", "pat-003": "src-003"}
    transform_allergies(allergies_csv, source_map, ctx)
    rows = ctx.output["dw.federated_allergies_v"]
    by_src = {r["source_id"]: r for r in rows}
    assert by_src["src-001"]["type"] == "Drug allergy"
    assert by_src["src-003"]["type"] == "Food allergy"


def test_allergies_transformer_carries_severity_and_reaction(allergies_csv, ctx):
    """Severity is normalized from Synthea's UPPERCASE to title case, the
    canonical warehouse form per the eval set (None / Mild / Moderate / Severe)."""
    source_map = {"pat-001": "src-001", "pat-003": "src-003"}
    transform_allergies(allergies_csv, source_map, ctx)
    rows = ctx.output["dw.federated_allergies_v"]
    by_src = {r["source_id"]: r for r in rows}
    assert by_src["src-001"]["severity"] == "Moderate"
    assert by_src["src-001"]["reaction"] == "Hives"


def test_allergies_transformer_carries_snomed_code(allergies_csv, ctx):
    """code is preserved verbatim. code_type is noise-injected (5% empty,
    rest are SNOMED variants) — verified separately via aggregate sampling."""
    source_map = {"pat-001": "src-001", "pat-003": "src-003"}
    transform_allergies(allergies_csv, source_map, ctx)
    rows = ctx.output["dw.federated_allergies_v"]
    by_src = {r["source_id"]: r for r in rows}
    assert by_src["src-001"]["code"] == "91936005"
    assert by_src["src-003"]["code"] == "91934008"


def test_allergies_transformer_emits_snomed_code_type_at_expected_rate(ctx):
    """Across many rows, the noise-injected code_type should resolve to a
    recognized SNOMED form most of the time. The SNOMED variant set
    includes an OID (2.16.840.1.113883.6.96) which is ~20% of picks and
    doesn't contain the substring "SNOMED" — so the floor is ~75%, not 95%."""
    import pandas as pd

    rows_in = pd.DataFrame(
        [
            {
                "START": "2020-01-01",
                "STOP": None,
                "PATIENT": f"pat-{i:03d}",
                "ENCOUNTER": f"enc-{i:03d}",
                "CODE": "91936005",
                "SYSTEM": "SNOMED-CT",
                "DESCRIPTION": "Allergy to penicillin",
                "TYPE": "allergy",
                "CATEGORY": "medication",
                "REACTION1": "247472004",
                "DESCRIPTION1": "Hives",
                "SEVERITY1": "MODERATE",
                "REACTION2": None,
                "DESCRIPTION2": None,
                "SEVERITY2": None,
            }
            for i in range(100)
        ]
    )
    source_map = {f"pat-{i:03d}": f"src-{i:03d}" for i in range(100)}
    transform_allergies(rows_in, source_map, ctx)
    rows = ctx.output["dw.federated_allergies_v"]
    snomed_rows = [r for r in rows if r["code_type"] and "SNOMED" in r["code_type"].upper()]
    assert len(snomed_rows) >= 65


def test_allergies_transformer_handles_empty_input(ctx):
    """No allergies CSV rows → empty list under the expected key (so the
    table still ships in the dump)."""
    import pandas as pd

    empty = pd.DataFrame(
        columns=[
            "START",
            "STOP",
            "PATIENT",
            "ENCOUNTER",
            "CODE",
            "SYSTEM",
            "DESCRIPTION",
            "TYPE",
            "CATEGORY",
            "REACTION1",
            "DESCRIPTION1",
            "SEVERITY1",
            "REACTION2",
            "DESCRIPTION2",
            "SEVERITY2",
        ]
    )
    transform_allergies(empty, {}, ctx)
    assert ctx.output.get("dw.federated_allergies_v", []) == []
