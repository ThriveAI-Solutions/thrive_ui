"""Load the committed sample dump into Postgres or SQLite.

Usage:
    uv run python scripts/load_sample_db.py                     # Postgres (default)
    uv run python scripts/load_sample_db.py --target sqlite --path ./sample.db
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from pathlib import Path

import zstandard as zstd

DEFAULT_DUMP = Path("data/sample/thrive_sample.sql.zst")


def _read_dump(dump_path: Path) -> str:
    raw = dump_path.read_bytes()
    dctx = zstd.ZstdDecompressor()
    return dctx.decompress(raw, max_output_size=200 * 1024 * 1024).decode("utf-8")


def _quote(v: str) -> str:
    """Convert a COPY-format text value to a SQL literal."""
    if v == "\\N":
        return "NULL"
    # Reverse the dump-writer escaping.
    unescaped = v.replace("\\t", "\t").replace("\\n", "\n").replace("\\r", "\r").replace("\\\\", "\\")
    if unescaped.lstrip("-").replace(".", "", 1).isdigit():
        return unescaped
    return "'" + unescaped.replace("'", "''") + "'"


def copy_blocks_to_inserts(sql: str, *, strip_schema: bool) -> str:
    """Replace every COPY ... FROM stdin / \\. block with INSERT statements."""
    pattern = re.compile(
        r"COPY ([\w.]+) \(([^)]+)\) FROM stdin;\n(.*?)\n\\\.\n",
        re.DOTALL,
    )

    def replace(m: re.Match) -> str:
        table = m.group(1)
        if strip_schema and "." in table:
            table = table.split(".", 1)[1]
        cols = m.group(2)
        body = m.group(3)
        values = []
        for line in body.split("\n"):
            if not line:
                continue
            fields = line.split("\t")
            values.append("(" + ", ".join(_quote(f) for f in fields) + ")")
        if not values:
            return ""
        # Batch in chunks of 500 rows for SQLite.
        chunks = [values[i : i + 500] for i in range(0, len(values), 500)]
        return "\n".join(f"INSERT INTO {table} ({cols}) VALUES\n  " + ",\n  ".join(chunk) + ";\n" for chunk in chunks)

    return pattern.sub(replace, sql)


def load_into_sqlite(dump_path: Path, db_path: Path) -> None:
    sql = _read_dump(dump_path)
    # Strip Postgres-only preamble (SET ..., CREATE SCHEMA), drop `CASCADE`
    # from DROP statements (not supported by SQLite), strip dw. prefixes.
    sql = re.sub(r"^SET [^;]+;\n", "", sql, flags=re.MULTILINE)
    sql = re.sub(r"CREATE SCHEMA[^;]+;\n", "", sql)
    sql = re.sub(r"\s+CASCADE(?=\s*;)", "", sql)
    sql = sql.replace("dw.", "")
    sql = copy_blocks_to_inserts(sql, strip_schema=True)
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    conn.executescript(sql)
    conn.commit()
    conn.close()


def load_into_postgres(dump_path: Path) -> None:
    """Load via psycopg2. Reads connection params from secrets.toml.

    The dump is a pg_dump-style text file: DDL statements followed by
    `COPY table (cols) FROM stdin;` blocks terminated by `\\.`. We split
    the text on the COPY boundaries, run DDL via plain cursor.execute,
    and feed each COPY body through cursor.copy_expert (which natively
    speaks the COPY text protocol).
    """
    import io
    import tomllib

    import psycopg2

    secrets_path = Path(".streamlit/secrets.toml")
    if not secrets_path.exists():
        print("ERROR: .streamlit/secrets.toml not found", file=sys.stderr)
        sys.exit(1)
    pg = tomllib.loads(secrets_path.read_text())["postgres"]
    sql = _read_dump(dump_path)

    conn = psycopg2.connect(
        host=pg["host"],
        port=pg["port"],
        user=pg["user"],
        password=pg["password"],
        dbname=pg["database"],
    )
    conn.autocommit = False
    pattern = re.compile(
        r"COPY ([\w.]+) \(([^)]+)\) FROM stdin;\n(.*?)\n\\\.\n",
        re.DOTALL,
    )
    try:
        cur = conn.cursor()
        last_end = 0
        for m in pattern.finditer(sql):
            ddl_chunk = sql[last_end : m.start()].strip()
            if ddl_chunk:
                cur.execute(ddl_chunk)
            table, cols, body = m.group(1), m.group(2), m.group(3)
            copy_sql = f"COPY {table} ({cols}) FROM stdin"
            cur.copy_expert(copy_sql, io.StringIO(body + "\n"))
            last_end = m.end()
        tail = sql[last_end:].strip()
        if tail:
            cur.execute(tail)
        conn.commit()
        print(f"Loaded {dump_path} → postgres://{pg['host']}:{pg['port']}/{pg['database']}")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["postgres", "sqlite"], default="postgres")
    parser.add_argument("--dump", type=Path, default=DEFAULT_DUMP)
    parser.add_argument("--path", type=Path, help="SQLite DB path (sqlite only)")
    args = parser.parse_args(argv)

    if not args.dump.exists():
        print(f"ERROR: {args.dump} not found", file=sys.stderr)
        return 1

    if args.target == "postgres":
        load_into_postgres(args.dump)
    else:
        if not args.path:
            args.path = Path("sample.db")
        load_into_sqlite(args.dump, args.path)
        print(f"Loaded {args.dump} → {args.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
