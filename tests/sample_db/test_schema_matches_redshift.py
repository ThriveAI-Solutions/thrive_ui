"""Schema contract test: dw.* tables in our DDL must match the Redshift catalog."""

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CATALOG = REPO / "docs/superpowers/research/2026-05-06-redshift-tables.json"
SCHEMA = REPO / "scripts/sample_db/schema.sql"

WHITELIST = [
    "internal_patient_profile_v",
    "internal_source_reference_v",
    "federated_demographic_v",
    "federated_demographic_history_v",
    "federated_encounters_v",
    "federated_problems_v",
    "federated_results_v",
    "federated_meds_v",
    "federated_orders_v",
    "federated_vaccination_v",
    "federated_vitals_v",
    "federated_documents_v",
    "federated_adt_v",
    "federated_claims_icd_diagnosis_detail_v",
    "federated_claims_icd_procedure_detail_v",
    "federated_claims_medical_facility_detail_v",
    "federated_claims_summary_v",
    "metric_federated_data_v",
]


def _parse_ddl_columns(ddl: str) -> dict[str, list[str]]:
    """Return {table_name: [col_name, ...]} from CREATE TABLE blocks."""
    out = {}
    pattern = re.compile(
        r"CREATE TABLE dw\.(\w+) \(\n(.*?)\n\);",
        re.DOTALL,
    )
    for m in pattern.finditer(ddl):
        table = m.group(1)
        body = m.group(2)
        col_names = [line.strip().split()[0] for line in body.split(",\n") if line.strip()]
        out[table] = col_names
    return out


def _catalog_columns() -> dict[str, list[str]]:
    rows = json.loads(CATALOG.read_text())
    out: dict[str, list[tuple[int, str]]] = {}
    for r in rows:
        v = r["viewname"]
        if v in WHITELIST:
            out.setdefault(v, []).append((r["column_position"], r["column_name"]))
    return {v: [c for _, c in sorted(out[v])] for v in out}


@pytest.mark.parametrize("table", WHITELIST)
def test_table_columns_match_redshift_catalog(table: str):
    ddl_cols = _parse_ddl_columns(SCHEMA.read_text())
    cat_cols = _catalog_columns()
    assert table in ddl_cols, f"{table} missing from schema.sql"
    assert table in cat_cols, (
        f"{table} missing from catalog JSON — refresh docs/superpowers/research/2026-05-06-redshift-tables.json"
    )
    assert ddl_cols[table] == cat_cols[table], (
        f"Column drift in {table}.\n  DDL:     {ddl_cols[table]}\n  Catalog: {cat_cols[table]}"
    )


def test_all_whitelist_tables_present():
    ddl_cols = _parse_ddl_columns(SCHEMA.read_text())
    assert set(ddl_cols.keys()) == set(WHITELIST)
