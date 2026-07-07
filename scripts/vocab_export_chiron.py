"""Export chiron's loaded vocabulary tables into a portable JSONL(+zstd) dump.

Standalone — plain SQLAlchemy, NO imports from the chiron repo. This lets the
script run from thrive_ui alone (chiron isn't a dependency here); Task 6 runs
it against chiron's real Postgres to produce the dump that
`scripts/import_vocab_dump.py` loads into the app DB.

Streams each table with a server-side cursor (`vocab_synonyms` and
`vocab_code_set_members` run to millions of rows) so the whole table is never
materialized in memory.

Run:
    uv run python scripts/vocab_export_chiron.py [--out data/vocab] [--db-url ...]

Env:
    CHIRON_APP_DB_URL   chiron's app Postgres connection string
                        (default: postgresql+psycopg2://postgres:postgres@localhost:5470/postgres)

Re-running overwrites the contents of --out.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from decimal import Decimal
from pathlib import Path

import zstandard as zstd
from sqlalchemy import MetaData, Table, create_engine, func, select

DEFAULT_DB_URL = "postgresql+psycopg2://postgres:postgres@localhost:5470/postgres"
DEFAULT_OUT = Path("data/vocab")

# Parents first — order doesn't matter for export (no FK-ordering constraint
# on a read-only SELECT), but keeping it stable makes the dump dir's manifest
# easier to eyeball against scripts/import_vocab_dump.py's insert order.
TABLES = [
    "vocab_codes",
    "vocab_code_sets",
    "vocab_synonyms",
    "vocab_code_set_members",
    "vocab_set_synonyms",
]

CHUNK_SIZE = 5000


def _json_default(value):
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _export_table_rows(conn, table: Table, jsonl_path: Path) -> int:
    """Stream `table` to `jsonl_path`, one row-dict per line. Returns row count."""
    row_count = 0
    result = conn.execution_options(yield_per=CHUNK_SIZE, stream_results=True).execute(select(table))
    with jsonl_path.open("w", encoding="utf-8") as f:
        for partition in result.partitions(CHUNK_SIZE):
            for row in partition:
                f.write(json.dumps(dict(row._mapping), default=_json_default))
                f.write("\n")
                row_count += 1
    return row_count


def _compress_and_remove(jsonl_path: Path, zst_path: Path) -> None:
    cctx = zstd.ZstdCompressor(level=19)
    with jsonl_path.open("rb") as src, zst_path.open("wb") as dst:
        cctx.copy_stream(src, dst)
    jsonl_path.unlink()


def export_all(db_url: str, out_dir: Path) -> dict:
    """Export all vocab tables from `db_url` into `out_dir`. Returns the manifest dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    engine = create_engine(db_url)
    metadata = MetaData()
    manifest: dict = {
        "exported_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "tables": {},
        "vocab_code_counts": {},
    }
    try:
        with engine.connect() as conn:
            reflected: dict[str, Table] = {}
            for table_name in TABLES:
                table = Table(table_name, metadata, autoload_with=engine)
                reflected[table_name] = table

                jsonl_path = out_dir / f"{table_name}.jsonl"
                zst_path = out_dir / f"{table_name}.jsonl.zst"
                row_count = _export_table_rows(conn, table, jsonl_path)
                _compress_and_remove(jsonl_path, zst_path)

                source_versions = None
                if "source_version" in table.c:
                    values = conn.execute(select(table.c.source_version).distinct()).scalars().all()
                    source_versions = sorted(v for v in values if v is not None)

                manifest["tables"][table_name] = {
                    "row_count": row_count,
                    "source_versions": source_versions,
                }

            codes_table = reflected["vocab_codes"]
            vocab_counts = conn.execute(
                select(codes_table.c.vocabulary, func.count()).group_by(codes_table.c.vocabulary)
            ).all()
            manifest["vocab_code_counts"] = {vocabulary: count for vocabulary, count in vocab_counts}
    finally:
        engine.dispose()

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output directory (default: data/vocab)")
    parser.add_argument(
        "--db-url",
        default=os.environ.get("CHIRON_APP_DB_URL", DEFAULT_DB_URL),
        help="chiron app Postgres URL (env CHIRON_APP_DB_URL overrides the default)",
    )
    args = parser.parse_args(argv)

    manifest = export_all(args.db_url, args.out)
    total_rows = sum(t["row_count"] for t in manifest["tables"].values())
    print(f"Exported {total_rows} rows across {len(TABLES)} tables to {args.out}")
    for table_name, info in manifest["tables"].items():
        print(f"  {table_name}: {info['row_count']} rows")
    print("Per-vocabulary code counts:")
    for vocabulary, count in sorted(manifest["vocab_code_counts"].items()):
        print(f"  {vocabulary}: {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
