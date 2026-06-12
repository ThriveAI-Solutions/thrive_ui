"""Synthea CSV → pg_dump-compatible SQL dump.

Usage:
    uv run python -m scripts.sample_db.etl \
        [--synthea-dir data/sample/synthea/csv] \
        [--out data/sample/thrive_sample.sql.zst] \
        [--seed 42]
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import pandas as pd
import zstandard as zstd

from scripts.sample_db.dump_writer import write_dump
from scripts.sample_db.transformers.base import TransformContext
from scripts.sample_db.transformers.allergies import transform_allergies
from scripts.sample_db.transformers.claims import transform_claims
from scripts.sample_db.transformers.adt import transform_adt
from scripts.sample_db.transformers.documents import transform_documents
from scripts.sample_db.transformers.encounters import transform_encounters
from scripts.sample_db.transformers.identity import transform_identity
from scripts.sample_db.transformers.immunizations import transform_immunizations
from scripts.sample_db.transformers.meds import transform_meds
from scripts.sample_db.transformers.orders import transform_orders
from scripts.sample_db.transformers.problems import transform_problems
from scripts.sample_db.transformers.results import transform_results
from scripts.sample_db.transformers.rollup import transform_rollup
from scripts.sample_db.transformers.vitals import transform_vitals


SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"

_REQUIRED_FILES = [
    "patients.csv",
    "encounters.csv",
    "conditions.csv",
    "observations.csv",
    "medications.csv",
    "procedures.csv",
    "immunizations.csv",
    "providers.csv",
    "organizations.csv",
    "claims.csv",
]

# Optional inputs — older Synthea outputs may pre-date these CSVs.
_OPTIONAL_FILES = [
    "allergies.csv",
]


def _parse_column_order(ddl: str) -> dict[str, list[str]]:
    import re

    out = {}
    pat = re.compile(r"CREATE TABLE (dw\.\w+) \(\n(.*?)\n\);", re.DOTALL)
    for m in pat.finditer(ddl):
        body = m.group(2)
        out[m.group(1)] = [line.strip().split()[0] for line in body.split(",\n") if line.strip()]
    return out


def _build_source_map(patients: pd.DataFrame) -> dict[str, str]:
    """Map Synthea patient ID → source_id."""
    return {p["Id"]: f"src-{p['Id']}" for _, p in patients.iterrows()}


def _build_patient_id_map(sources: list[dict]) -> dict[str, int]:
    """source_id → integer patient_id, excluding empi_rank=99 inactive rows."""
    return {s["source_id"]: s["patient_id"] for s in sources if s["empi_rank"] != 99}


def _build_practice_map(sources: list[dict]) -> dict[str, str]:
    return {s["source_id"]: s.get("source_name") or "Unknown" for s in sources}


def _build_claim_to_source(claims: pd.DataFrame, source_map: dict[str, str]) -> dict[str, str]:
    """Build claim_line_identifier → source_id replicating Task 18's ID scheme."""
    out = {}
    for line_seq, (_, c) in enumerate(claims.iterrows(), start=1):
        sid = source_map.get(c["PATIENTID"])
        if sid is None:
            continue
        out[f"{c['Id']}-L{line_seq:03d}"] = sid
    return out


def run_etl_in_memory(inputs: dict[str, pd.DataFrame], seed: int = 42) -> dict[str, list[dict]]:
    """Run all transformers on in-memory dataframes. Returns ctx.output."""
    ctx = TransformContext(seed=seed)
    patients = inputs["patients.csv"]
    encounters = inputs["encounters.csv"]
    conditions = inputs["conditions.csv"]
    observations = inputs["observations.csv"]
    medications = inputs["medications.csv"]
    procedures = inputs["procedures.csv"]
    immunizations = inputs["immunizations.csv"]
    claims = inputs["claims.csv"]

    transform_identity(patients, encounters, ctx)
    source_map = _build_source_map(patients)

    transform_encounters(encounters, source_map, ctx)
    transform_problems(conditions, source_map, ctx)
    transform_results(observations, source_map, ctx)
    transform_meds(medications, source_map, ctx)
    transform_orders(procedures, source_map, ctx)
    transform_immunizations(immunizations, source_map, ctx)
    transform_vitals(observations, source_map, ctx)
    transform_documents(encounters, source_map, ctx)
    transform_claims(claims, encounters, procedures, source_map, ctx)
    allergies = inputs.get("allergies.csv")
    if allergies is not None:
        transform_allergies(allergies, source_map, ctx)

    # ADT events derive from inpatient/emergency encounters and need the
    # integer patient_id (federated_adt_v's only identity column). Build
    # the synthea_id → patient_id map from the source-reference output and
    # run AFTER transform_identity so the input is populated.
    src_to_pid_for_adt = _build_patient_id_map(ctx.output["dw.internal_source_reference_v"])
    synthea_to_pid = {
        syn_id: src_to_pid_for_adt[src_id] for syn_id, src_id in source_map.items() if src_id in src_to_pid_for_adt
    }
    transform_adt(encounters, synthea_to_pid, ctx)

    # Build rollup auxiliary maps and call.
    src_to_pid = _build_patient_id_map(ctx.output["dw.internal_source_reference_v"])
    src_to_practice = _build_practice_map(ctx.output["dw.internal_source_reference_v"])
    claim_to_source = _build_claim_to_source(claims, source_map)
    transform_rollup(src_to_pid, src_to_practice, ctx, claim_to_source=claim_to_source)

    # Ensure all expected tables exist (some may be empty).
    for t in _expected_tables():
        ctx.output.setdefault(t, [])
    return ctx.output


def _expected_tables() -> list[str]:
    return [
        f"dw.{v}"
        for v in [
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
            "federated_allergies_v",
            "federated_claims_icd_diagnosis_detail_v",
            "federated_claims_icd_procedure_detail_v",
            "federated_claims_medical_facility_detail_v",
            "federated_claims_summary_v",
            "metric_federated_data_v",
        ]
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthea-dir", type=Path, default=Path("data/sample/synthea/csv"))
    parser.add_argument("--out", type=Path, default=Path("data/sample/thrive_sample.sql.zst"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    if not args.synthea_dir.exists():
        print(f"ERROR: {args.synthea_dir} not found. Run scripts/synthea/generate.sh first.", file=sys.stderr)
        return 1

    inputs = {}
    for f in _REQUIRED_FILES:
        path = args.synthea_dir / f
        if not path.exists():
            print(f"ERROR: missing {path}", file=sys.stderr)
            return 1
        inputs[f] = pd.read_csv(path)
    for f in _OPTIONAL_FILES:
        path = args.synthea_dir / f
        if path.exists():
            inputs[f] = pd.read_csv(path)
        else:
            print(f"NOTE: {path} not found — skipping (table will be empty in dump)")

    print(f"Running ETL on {sum(len(d) for d in inputs.values())} input rows...")
    output = run_etl_in_memory(inputs, seed=args.seed)

    # Get column order from schema.sql.
    ddl = SCHEMA_PATH.read_text()
    col_order = _parse_column_order(ddl)

    # Sort each table's rows by serialized form for byte-determinism.
    table_data: dict[str, tuple[list[str], list[dict]]] = {}
    for table, rows in output.items():
        cols = col_order[table]
        sorted_rows = sorted(rows, key=lambda r: tuple(str(r.get(c) or "") for c in cols))
        table_data[table] = (cols, sorted_rows)

    buf = io.StringIO()
    write_dump(buf, ddl, table_data)
    sql_text = buf.getvalue()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cctx = zstd.ZstdCompressor(level=19)
    with args.out.open("wb") as fh:
        fh.write(cctx.compress(sql_text.encode("utf-8")))

    print(f"Wrote {args.out} ({args.out.stat().st_size:,} bytes, {len(sql_text):,} uncompressed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
