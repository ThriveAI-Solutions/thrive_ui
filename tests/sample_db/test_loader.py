"""Loader unit tests — SQLite path is hermetic; Postgres path needs Docker."""

import sqlite3
from pathlib import Path

import pytest
import zstandard as zstd

from scripts.load_sample_db import (
    copy_blocks_to_inserts,
    load_into_sqlite,
)
from scripts.sample_db.dump_writer import schema_sha256


REPO = Path(__file__).resolve().parents[2]
COMMITTED_DUMP = REPO / "data/sample/thrive_sample.sql.zst"
SAMPLE_SCHEMA = REPO / "scripts/sample_db/schema.sql"


@pytest.fixture
def tiny_dump(tmp_path: Path) -> Path:
    """A minimal dump with one table and two rows."""
    sql = """-- test dump
CREATE SCHEMA IF NOT EXISTS dw;
CREATE TABLE dw.t (id INTEGER, name VARCHAR(50));

COPY dw.t (id, name) FROM stdin;
1\tAlice
2\tBob
\\.

"""
    cctx = zstd.ZstdCompressor()
    compressed = cctx.compress(sql.encode("utf-8"))
    p = tmp_path / "test.sql.zst"
    p.write_bytes(compressed)
    return p


def test_copy_blocks_translated_to_inserts():
    sql = "COPY dw.t (id, name) FROM stdin;\n1\tAlice\n2\tBob\n\\.\n"
    out = copy_blocks_to_inserts(sql, strip_schema=True)
    assert "INSERT INTO t (id, name) VALUES" in out
    assert "(1, 'Alice')" in out
    assert "(2, 'Bob')" in out
    assert "COPY" not in out


def test_copy_block_handles_nulls():
    sql = "COPY dw.t (id, name) FROM stdin;\n1\t\\N\n\\.\n"
    out = copy_blocks_to_inserts(sql, strip_schema=True)
    assert "(1, NULL)" in out


def test_load_into_sqlite(tmp_path: Path, tiny_dump: Path):
    db_path = tmp_path / "sample.db"
    load_into_sqlite(tiny_dump, db_path)
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT id, name FROM t ORDER BY id").fetchall()
    assert rows == [(1, "Alice"), (2, "Bob")]


def test_committed_dump_matches_current_sample_schema():
    """Schema changes must ship with a regenerated synthetic dump."""
    expected = schema_sha256(SAMPLE_SCHEMA.read_text())
    with COMMITTED_DUMP.open("rb") as compressed:
        with zstd.ZstdDecompressor().stream_reader(compressed) as reader:
            dump_prefix = reader.read(256 * 1024).decode("utf-8")

    assert f"VALUES ('{expected}')" in dump_prefix, (
        "Committed sample dump is stale — run 'uv run python -m scripts.sample_db.etl'"
    )
