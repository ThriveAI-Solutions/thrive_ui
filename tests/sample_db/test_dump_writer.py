import io

from scripts.sample_db.dump_writer import write_copy_block, write_dump


def test_copy_block_emits_pg_dump_format():
    buf = io.StringIO()
    rows = [
        {"id": 1, "name": "Alice", "age": 30},
        {"id": 2, "name": "Bob", "age": None},
    ]
    write_copy_block(buf, "dw.users", ["id", "name", "age"], rows)
    out = buf.getvalue()
    assert "COPY dw.users (id, name, age) FROM stdin;" in out
    assert "1\tAlice\t30" in out
    assert "2\tBob\t\\N" in out  # NULL serialization
    assert out.rstrip().endswith("\\.")


def test_copy_block_escapes_tabs_and_newlines():
    buf = io.StringIO()
    rows = [{"id": 1, "note": "line1\nline2\twith\ttabs"}]
    write_copy_block(buf, "dw.docs", ["id", "note"], rows)
    out = buf.getvalue()
    assert "1\tline1\\nline2\\twith\\ttabs" in out


def test_write_dump_concatenates_ddl_and_data():
    ddl = "CREATE TABLE dw.t (id INTEGER);"
    table_data = {"dw.t": (["id"], [{"id": 1}, {"id": 2}])}
    buf = io.StringIO()
    write_dump(buf, ddl, table_data)
    out = buf.getvalue()
    assert "CREATE TABLE dw.t" in out
    assert "COPY dw.t (id) FROM stdin;" in out
    assert "1" in out and "2" in out


def test_write_dump_is_deterministic_given_same_input():
    ddl = "CREATE TABLE dw.t (id INTEGER);"
    table_data = {"dw.t": (["id"], [{"id": i} for i in range(50)])}
    a = io.StringIO()
    write_dump(a, ddl, table_data)
    b = io.StringIO()
    write_dump(b, ddl, table_data)
    assert a.getvalue() == b.getvalue()
