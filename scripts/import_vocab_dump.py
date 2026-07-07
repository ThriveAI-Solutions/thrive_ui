"""Import a chiron vocab dump (see scripts/vocab_export_chiron.py) into the app DB.

Resolves the app DB the same way the rest of the app does — via
`orm.models.engine` (SQLite path from `THRIVE_SQLITE_PATH` env or
`.streamlit/secrets.toml`, per `orm/models.py::_get_database_url`).

Requires the vocab_* tables to already exist (created by the Task 1 Alembic
migration) — run `uv run alembic upgrade head` first. This script never
creates tables itself.

Delete-then-insert, all in ONE transaction: child tables are cleared first,
then rows are (re)inserted parents-first with their original IDs preserved
from the dump (vocab_codes.id and vocab_code_sets.set_id are FK targets, so
identity must round-trip exactly). Before commit, a battery of sanity floors
runs inside the same transaction; any failure rolls back the whole import and
exits non-zero, leaving the DB exactly as it was.

Run:
    uv run python scripts/import_vocab_dump.py [dump_dir] [--allow-partial] [--vacuum]

dump_dir defaults to data/vocab/.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Iterator

import zstandard as zstd
from sqlalchemy import Engine, Table, delete, func, insert, inspect, select, text

# sys.path shim — same pattern as scripts/seed_agent_rag.py — so `from orm
# import models` resolves when this script is run directly (not as `python -m`).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Module-level so tests can monkeypatch them down and use a tiny synthetic dump.
FLOOR_VOCAB_CODES = 50_000
FLOOR_CODE_SET_MEMBERS = 50_000
FLOOR_CODE_SETS = 100
REQUIRED_VOCABULARIES = ("icd10", "icd9", "loinc", "cvx", "rxnorm", "cpt", "snomed")

CHUNK_SIZE = 5000

# Child-first delete order, parent-first insert order (see module docstring).
DELETE_ORDER = [
    "vocab_set_synonyms",
    "vocab_code_set_members",
    "vocab_synonyms",
    "vocab_codes",
    "vocab_code_sets",
]
INSERT_ORDER = [
    "vocab_codes",
    "vocab_code_sets",
    "vocab_synonyms",
    "vocab_code_set_members",
    "vocab_set_synonyms",
]


class VocabImportError(Exception):
    """Raised when a dump fails a precondition or a post-load sanity floor."""


def _model_tables() -> dict[str, Table]:
    # Imported lazily so importing this module doesn't require orm.models
    # (and therefore streamlit/secrets) unless it's actually needed.
    from orm.models import VocabCode, VocabCodeSet, VocabCodeSetMember, VocabSetSynonym, VocabSynonym

    return {
        "vocab_codes": VocabCode.__table__,
        "vocab_code_sets": VocabCodeSet.__table__,
        "vocab_synonyms": VocabSynonym.__table__,
        "vocab_code_set_members": VocabCodeSetMember.__table__,
        "vocab_set_synonyms": VocabSetSynonym.__table__,
    }


def _default_engine() -> Engine:
    from orm.models import engine

    return engine


def _resolve_dump_file(dump_dir: Path, table_name: str) -> Path:
    zst_path = dump_dir / f"{table_name}.jsonl.zst"
    plain_path = dump_dir / f"{table_name}.jsonl"
    if zst_path.exists():
        return zst_path
    if plain_path.exists():
        return plain_path
    raise VocabImportError(f"missing dump file for table '{table_name}' in {dump_dir} (looked for {zst_path.name})")


def _iter_rows(path: Path) -> Iterator[dict]:
    if path.suffix == ".zst":
        dctx = zstd.ZstdDecompressor()
        with path.open("rb") as fh, dctx.stream_reader(fh) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")
            for line in text_stream:
                line = line.strip()
                if line:
                    yield json.loads(line)
    else:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)


def _insert_rows(conn, table: Table, path: Path) -> int:
    count = 0
    chunk: list[dict] = []
    for row in _iter_rows(path):
        chunk.append(row)
        if len(chunk) >= CHUNK_SIZE:
            conn.execute(insert(table), chunk)
            count += len(chunk)
            chunk = []
    if chunk:
        conn.execute(insert(table), chunk)
        count += len(chunk)
    return count


def _table_count(conn, table: Table) -> int:
    return conn.execute(select(func.count()).select_from(table)).scalar_one()


def _orphan_count(conn, child: Table, child_fk_col: str, parent: Table, parent_pk_col: str) -> int:
    join_cond = child.c[child_fk_col] == parent.c[parent_pk_col]
    return conn.execute(
        select(func.count()).select_from(child.outerjoin(parent, join_cond)).where(parent.c[parent_pk_col].is_(None))
    ).scalar_one()


def _run_floor_checks(
    conn, tables: dict[str, Table], loaded_counts: dict[str, int], manifest: dict, *, allow_partial: bool
) -> None:
    failures: list[str] = []
    warnings: list[str] = []

    codes_count = _table_count(conn, tables["vocab_codes"])
    members_count = _table_count(conn, tables["vocab_code_set_members"])
    sets_count = _table_count(conn, tables["vocab_code_sets"])

    if codes_count < FLOOR_VOCAB_CODES:
        failures.append(f"vocab_codes has {codes_count} rows, below floor {FLOOR_VOCAB_CODES}")
    if members_count < FLOOR_CODE_SET_MEMBERS:
        failures.append(f"vocab_code_set_members has {members_count} rows, below floor {FLOOR_CODE_SET_MEMBERS}")
    if sets_count < FLOOR_CODE_SETS:
        failures.append(f"vocab_code_sets has {sets_count} rows, below floor {FLOOR_CODE_SETS}")

    vocab_counts = dict(
        conn.execute(
            select(tables["vocab_codes"].c.vocabulary, func.count()).group_by(tables["vocab_codes"].c.vocabulary)
        ).all()
    )
    for vocabulary in REQUIRED_VOCABULARIES:
        if vocab_counts.get(vocabulary, 0) < 1:
            msg = f"vocabulary '{vocabulary}' has zero vocab_codes rows"
            (warnings if allow_partial else failures).append(msg)

    for table_name, expected in manifest["tables"].items():
        actual = loaded_counts.get(table_name)
        expected_count = expected["row_count"]
        if actual != expected_count:
            failures.append(f"{table_name}: loaded {actual} rows, manifest says {expected_count}")

    orphan_checks = [
        ("vocab_code_set_members.code_id", tables["vocab_code_set_members"], "code_id", tables["vocab_codes"], "id"),
        (
            "vocab_code_set_members.set_id",
            tables["vocab_code_set_members"],
            "set_id",
            tables["vocab_code_sets"],
            "set_id",
        ),
        ("vocab_synonyms.code_id", tables["vocab_synonyms"], "code_id", tables["vocab_codes"], "id"),
        ("vocab_set_synonyms.set_id", tables["vocab_set_synonyms"], "set_id", tables["vocab_code_sets"], "set_id"),
    ]
    for label, child, fk_col, parent, pk_col in orphan_checks:
        orphans = _orphan_count(conn, child, fk_col, parent, pk_col)
        if orphans:
            failures.append(f"{label}: {orphans} orphaned row(s)")

    if failures:
        raise VocabImportError("vocab import failed sanity checks:\n  " + "\n  ".join(failures))
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)


def _print_summary(conn, tables: dict[str, Table]) -> None:
    vocab_counts = conn.execute(
        select(tables["vocab_codes"].c.vocabulary, func.count()).group_by(tables["vocab_codes"].c.vocabulary)
    ).all()
    print("Per-vocabulary code counts:")
    for vocabulary, count in sorted(vocab_counts):
        print(f"  {vocabulary}: {count}")
    print("Totals:")
    print(f"  vocab_code_sets: {_table_count(conn, tables['vocab_code_sets'])}")
    print(f"  vocab_code_set_members: {_table_count(conn, tables['vocab_code_set_members'])}")
    print(f"  vocab_synonyms: {_table_count(conn, tables['vocab_synonyms'])}")
    print(f"  vocab_set_synonyms: {_table_count(conn, tables['vocab_set_synonyms'])}")


def import_dump(
    dump_dir: Path,
    engine: Engine | None = None,
    *,
    allow_partial: bool = False,
    vacuum: bool = False,
) -> None:
    """Load `dump_dir` into `engine` (defaults to orm.models.engine).

    Raises VocabImportError on any precondition/sanity-floor failure. The
    delete+insert happens in one transaction, so a raised error leaves the
    DB exactly as it was before the call.
    """
    dump_dir = Path(dump_dir)
    if engine is None:
        engine = _default_engine()

    if not inspect(engine).has_table("vocab_codes"):
        raise VocabImportError(
            "vocab_codes table does not exist — run `uv run alembic upgrade head` first "
            "(this script does not create tables)."
        )

    manifest_path = dump_dir / "manifest.json"
    if not manifest_path.exists():
        raise VocabImportError(f"manifest.json not found in {dump_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    tables = _model_tables()

    # Resolve every dump file up front so a missing file aborts before we
    # touch the DB at all.
    dump_files = {table_name: _resolve_dump_file(dump_dir, table_name) for table_name in INSERT_ORDER}

    with engine.begin() as conn:
        for table_name in DELETE_ORDER:
            conn.execute(delete(tables[table_name]))

        loaded_counts: dict[str, int] = {}
        for table_name in INSERT_ORDER:
            loaded_counts[table_name] = _insert_rows(conn, tables[table_name], dump_files[table_name])

        _run_floor_checks(conn, tables, loaded_counts, manifest, allow_partial=allow_partial)
        _print_summary(conn, tables)

    if vacuum and engine.dialect.name == "sqlite":
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            conn.execute(text("VACUUM"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dump_dir", nargs="?", type=Path, default=Path("data/vocab"))
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Downgrade the per-vocabulary presence floor to a warning (e.g. LOINC legitimately absent).",
    )
    parser.add_argument("--vacuum", action="store_true", help="Run VACUUM after commit (SQLite only).")
    args = parser.parse_args(argv)

    try:
        import_dump(args.dump_dir, allow_partial=args.allow_partial, vacuum=args.vacuum)
    except VocabImportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
