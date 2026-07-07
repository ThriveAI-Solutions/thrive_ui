"""Tests for scripts/import_vocab_dump.py.

Uses a tmp_path SQLite DB (tables created via Base.metadata.create_all) plus
a tiny synthetic on-disk dump — nothing here touches the real app DB or a
real chiron export. Floor constants are monkeypatched down (they're
module-level for exactly this reason) so the synthetic dump can stay tiny.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import zstandard as zstd
from sqlalchemy import create_engine, func, select

from orm.models import (
    Base,
    VocabCode,
    VocabCodeSet,
    VocabCodeSetMember,
    VocabSetSynonym,
    VocabSynonym,
)
from scripts import import_vocab_dump as ivd

REQUIRED_VOCABULARIES = ivd.REQUIRED_VOCABULARIES


def _codes_rows() -> list[dict]:
    return [
        {
            "id": i + 1,
            "vocabulary": vocab,
            "code": f"C{i + 1}",
            "code_norm": f"C{i + 1}",
            "display": f"Display {i + 1}",
            "is_active": True,
            "source_version": "v1",
        }
        for i, vocab in enumerate(REQUIRED_VOCABULARIES)
    ]


def _code_set_rows() -> list[dict]:
    return [
        {"set_id": "dx:test-1", "name": "Test Set 1", "source": "curated", "source_version": "v1"},
        {"set_id": "dx:test-2", "name": "Test Set 2", "source": "curated", "source_version": "v1"},
    ]


def _synonym_rows() -> list[dict]:
    return [
        {"id": 1, "code_id": 1, "term": "Diabetes", "term_norm": "diabetes", "source": "curated", "is_lay": True},
    ]


def _member_rows() -> list[dict]:
    # Only reference code_id 1 (icd10) and 2 (icd9) so the allow-partial test
    # (which drops the loinc row, id 3) doesn't incidentally orphan a member.
    return [
        {"id": 1, "set_id": "dx:test-1", "code_id": 1},
        {"id": 2, "set_id": "dx:test-1", "code_id": 2},
        {"id": 3, "set_id": "dx:test-2", "code_id": 2},
    ]


def _set_synonym_rows() -> list[dict]:
    return [
        {"id": 1, "set_id": "dx:test-1", "term": "DX1", "term_norm": "dx1"},
    ]


def _default_tables_data() -> dict[str, list[dict]]:
    return {
        "vocab_codes": _codes_rows(),
        "vocab_code_sets": _code_set_rows(),
        "vocab_synonyms": _synonym_rows(),
        "vocab_code_set_members": _member_rows(),
        "vocab_set_synonyms": _set_synonym_rows(),
    }


def _write_dump(
    dump_dir: Path,
    tables_data: dict[str, list[dict]],
    *,
    compress: bool,
    manifest_counts: dict[str, int] | None = None,
) -> None:
    dump_dir.mkdir(parents=True, exist_ok=True)
    for table_name, rows in tables_data.items():
        text = "\n".join(json.dumps(row) for row in rows)
        if rows:
            text += "\n"
        if compress:
            cctx = zstd.ZstdCompressor(level=3)
            (dump_dir / f"{table_name}.jsonl.zst").write_bytes(cctx.compress(text.encode("utf-8")))
        else:
            (dump_dir / f"{table_name}.jsonl").write_text(text, encoding="utf-8")

    counts = manifest_counts or {name: len(rows) for name, rows in tables_data.items()}
    manifest = {
        "exported_at": "2026-01-01T00:00:00+00:00",
        "tables": {name: {"row_count": count, "source_versions": ["v1"]} for name, count in counts.items()},
        "vocab_code_counts": {},
    }
    (dump_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


@pytest.fixture()
def engine(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path / 'app.sqlite3'}")
    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture(autouse=True)
def _lowered_floors(monkeypatch):
    """Lower the module-level floor constants so tiny synthetic dumps pass them."""
    monkeypatch.setattr(ivd, "FLOOR_VOCAB_CODES", 5)
    monkeypatch.setattr(ivd, "FLOOR_CODE_SET_MEMBERS", 2)
    monkeypatch.setattr(ivd, "FLOOR_CODE_SETS", 2)


def test_round_trip_fidelity(tmp_path, engine):
    dump_dir = tmp_path / "dump"
    _write_dump(dump_dir, _default_tables_data(), compress=True)

    ivd.import_dump(dump_dir, engine=engine)

    with engine.connect() as conn:
        assert conn.execute(select(func.count()).select_from(VocabCode.__table__)).scalar_one() == 7
        assert conn.execute(select(func.count()).select_from(VocabCodeSet.__table__)).scalar_one() == 2
        assert conn.execute(select(func.count()).select_from(VocabSynonym.__table__)).scalar_one() == 1
        assert conn.execute(select(func.count()).select_from(VocabCodeSetMember.__table__)).scalar_one() == 3
        assert conn.execute(select(func.count()).select_from(VocabSetSynonym.__table__)).scalar_one() == 1

        row = conn.execute(select(VocabCode.__table__).where(VocabCode.__table__.c.id == 1)).mappings().one()
        assert row["vocabulary"] == "icd10"
        assert row["display"] == "Display 1"
        assert row["id"] == 1  # PK preserved verbatim, not renumbered

        set_row = (
            conn.execute(select(VocabCodeSet.__table__).where(VocabCodeSet.__table__.c.set_id == "dx:test-1"))
            .mappings()
            .one()
        )
        assert set_row["name"] == "Test Set 1"


def test_precondition_missing_table(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path / 'no_tables.sqlite3'}")
    dump_dir = tmp_path / "dump"
    _write_dump(dump_dir, _default_tables_data(), compress=False)

    with pytest.raises(ivd.VocabImportError, match="alembic upgrade head"):
        ivd.import_dump(dump_dir, engine=eng)
    eng.dispose()


def test_floor_failure_rolls_back_and_preserves_existing_row(tmp_path, engine, monkeypatch):
    # Raise the floor back up so our tiny dump fails it.
    monkeypatch.setattr(ivd, "FLOOR_VOCAB_CODES", 1_000_000)

    # Pre-seed a row; delete-then-insert deletes it in-transaction, but a
    # failed floor check must roll the whole transaction back, so it must
    # still be there afterwards.
    with engine.begin() as conn:
        conn.execute(
            VocabCode.__table__.insert(),
            {
                "id": 999,
                "vocabulary": "icd10",
                "code": "SENTINEL",
                "code_norm": "SENTINEL",
                "display": "Pre-existing sentinel row",
                "is_active": True,
                "source_version": "pre",
            },
        )

    dump_dir = tmp_path / "dump"
    _write_dump(dump_dir, _default_tables_data(), compress=False)

    with pytest.raises(ivd.VocabImportError, match="below floor"):
        ivd.import_dump(dump_dir, engine=engine)

    with engine.connect() as conn:
        row = conn.execute(select(VocabCode.__table__).where(VocabCode.__table__.c.id == 999)).mappings().one_or_none()
    assert row is not None
    assert row["code"] == "SENTINEL"


def test_idempotent_reimport(tmp_path, engine):
    dump_dir = tmp_path / "dump"
    _write_dump(dump_dir, _default_tables_data(), compress=False)

    ivd.import_dump(dump_dir, engine=engine)
    with engine.connect() as conn:
        first_ids = sorted(r[0] for r in conn.execute(select(VocabCode.__table__.c.id)))
        first_count = conn.execute(select(func.count()).select_from(VocabCodeSetMember.__table__)).scalar_one()

    ivd.import_dump(dump_dir, engine=engine)  # second run must not raise or duplicate
    with engine.connect() as conn:
        second_ids = sorted(r[0] for r in conn.execute(select(VocabCode.__table__.c.id)))
        second_count = conn.execute(select(func.count()).select_from(VocabCodeSetMember.__table__)).scalar_one()

    assert first_ids == second_ids == [1, 2, 3, 4, 5, 6, 7]
    assert first_count == second_count == 3


def test_allow_partial_flag(tmp_path, engine):
    tables_data = _default_tables_data()
    # Drop the loinc row (REQUIRED_VOCABULARIES[2] -> id 3).
    tables_data["vocab_codes"] = [row for row in tables_data["vocab_codes"] if row["vocabulary"] != "loinc"]
    dump_dir = tmp_path / "dump"
    _write_dump(dump_dir, tables_data, compress=False)

    with pytest.raises(ivd.VocabImportError, match="loinc"):
        ivd.import_dump(dump_dir, engine=engine)

    # DB must be untouched after the failed attempt (rolled back).
    with engine.connect() as conn:
        assert conn.execute(select(func.count()).select_from(VocabCode.__table__)).scalar_one() == 0

    # Same dump succeeds with --allow-partial (downgrades to a warning).
    ivd.import_dump(dump_dir, engine=engine, allow_partial=True)
    with engine.connect() as conn:
        count = conn.execute(select(func.count()).select_from(VocabCode.__table__)).scalar_one()
    assert count == 6


def test_manifest_mismatch_aborts(tmp_path, engine):
    dump_dir = tmp_path / "dump"
    tables_data = _default_tables_data()
    # Manifest claims one more vocab_codes row than the dump file actually has.
    manifest_counts = {name: len(rows) for name, rows in tables_data.items()}
    manifest_counts["vocab_codes"] += 1
    _write_dump(dump_dir, tables_data, compress=False, manifest_counts=manifest_counts)

    with pytest.raises(ivd.VocabImportError, match="manifest says"):
        ivd.import_dump(dump_dir, engine=engine)

    with engine.connect() as conn:
        count = conn.execute(select(func.count()).select_from(VocabCode.__table__)).scalar_one()
    assert count == 0  # rolled back, DB left in its pre-existing (empty) state
