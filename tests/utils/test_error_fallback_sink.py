"""Tests for utils.error_fallback_sink (Phase 1: config + write_fallback_record)."""

from __future__ import annotations

import json
import logging
import logging.handlers as logging_handlers
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from orm.models import Base, ErrorCategory, ErrorLog, ErrorSeverity
from utils.error_fallback_sink import (
    ErrorLoggingConfig,
    _reset_handler_cache,
    read_fallback_records,
    try_drain_fallback_to_db,
    write_fallback_record,
)


@pytest.fixture(autouse=True)
def _reset_cache():
    _reset_handler_cache()
    yield
    _reset_handler_cache()


def _make_config(tmp_path: Path, **overrides) -> ErrorLoggingConfig:
    return ErrorLoggingConfig(
        fallback_path=tmp_path / "error_fallback.jsonl",
        fallback_max_bytes=overrides.get("fallback_max_bytes", 5_000_000),
        fallback_backup_count=overrides.get("fallback_backup_count", 5),
    )


# ── ErrorLoggingConfig ────────────────────────────────────────────────────


class TestErrorLoggingConfig:
    def test_defaults(self):
        cfg = ErrorLoggingConfig()
        assert cfg.fallback_path.name == "error_fallback.jsonl"
        assert cfg.fallback_path.parent.name == "logs"
        assert cfg.fallback_max_bytes == 5_000_000
        assert cfg.fallback_backup_count == 5

    def test_from_secrets_full_overrides(self, tmp_path):
        section = {
            "fallback_path": str(tmp_path / "custom.jsonl"),
            "fallback_max_bytes": 1024,
            "fallback_backup_count": 2,
        }
        cfg = ErrorLoggingConfig.from_secrets(section)
        assert cfg.fallback_path == tmp_path / "custom.jsonl"
        assert cfg.fallback_max_bytes == 1024
        assert cfg.fallback_backup_count == 2

    def test_from_secrets_partial_uses_defaults(self):
        defaults = ErrorLoggingConfig()
        cfg = ErrorLoggingConfig.from_secrets({"fallback_max_bytes": 999})
        assert cfg.fallback_max_bytes == 999
        assert cfg.fallback_path == defaults.fallback_path
        assert cfg.fallback_backup_count == defaults.fallback_backup_count

    def test_from_secrets_none_returns_defaults(self):
        cfg = ErrorLoggingConfig.from_secrets(None)
        assert cfg == ErrorLoggingConfig()

    def test_from_secrets_empty_dict_returns_defaults(self):
        cfg = ErrorLoggingConfig.from_secrets({})
        assert cfg == ErrorLoggingConfig()

    def test_from_secrets_invalid_int_coerces_to_default(self):
        defaults = ErrorLoggingConfig()
        cfg = ErrorLoggingConfig.from_secrets({"fallback_max_bytes": "not-a-number", "fallback_backup_count": None})
        assert cfg.fallback_max_bytes == defaults.fallback_max_bytes
        assert cfg.fallback_backup_count == defaults.fallback_backup_count

    def test_from_streamlit_missing_section_returns_defaults(self):
        with patch(
            "utils.error_fallback_sink._get_streamlit_section",
            return_value=None,
        ):
            cfg = ErrorLoggingConfig.from_streamlit()
        assert cfg == ErrorLoggingConfig()

    def test_from_streamlit_empty_section_returns_defaults(self):
        with patch(
            "utils.error_fallback_sink._get_streamlit_section",
            return_value={},
        ):
            cfg = ErrorLoggingConfig.from_streamlit()
        assert cfg == ErrorLoggingConfig()

    def test_from_streamlit_handles_import_failure(self):
        with patch(
            "utils.error_fallback_sink._get_streamlit_section",
            side_effect=RuntimeError("streamlit not available"),
        ):
            cfg = ErrorLoggingConfig.from_streamlit()
        assert cfg == ErrorLoggingConfig()


# ── write_fallback_record ─────────────────────────────────────────────────


class TestWriteFallbackRecord:
    def test_appends_one_line_json(self, tmp_path):
        cfg = _make_config(tmp_path)
        write_fallback_record({"a": 1, "b": "x"}, config=cfg)
        contents = cfg.fallback_path.read_text(encoding="utf-8")
        lines = [line for line in contents.splitlines() if line]
        assert len(lines) == 1
        assert json.loads(lines[0]) == {"a": 1, "b": "x"}

    def test_appends_multiple_lines_in_order(self, tmp_path):
        cfg = _make_config(tmp_path)
        for i in range(3):
            write_fallback_record({"i": i}, config=cfg)
        contents = cfg.fallback_path.read_text(encoding="utf-8")
        lines = [line for line in contents.splitlines() if line]
        assert [json.loads(line)["i"] for line in lines] == [0, 1, 2]

    def test_creates_parent_directory(self, tmp_path):
        nested = tmp_path / "deep" / "nested" / "logs" / "fallback.jsonl"
        cfg = ErrorLoggingConfig(
            fallback_path=nested,
            fallback_max_bytes=5_000_000,
            fallback_backup_count=5,
        )
        write_fallback_record({"k": "v"}, config=cfg)
        assert nested.exists()
        assert json.loads(nested.read_text(encoding="utf-8").strip()) == {"k": "v"}

    def test_handles_datetime_via_default_str(self, tmp_path):
        cfg = _make_config(tmp_path)
        when = datetime(2026, 6, 5, 12, 34, 56)
        write_fallback_record({"created_at": when}, config=cfg)
        line = cfg.fallback_path.read_text(encoding="utf-8").strip()
        payload = json.loads(line)
        assert payload["created_at"] == str(when)

    def test_defensive_on_serialize_failure(self, tmp_path):
        cfg = _make_config(tmp_path)

        class Boom:
            def __str__(self):
                raise RuntimeError("boom")

            def __repr__(self):
                raise RuntimeError("boom-repr")

        # Must not raise.
        write_fallback_record({"x": Boom()}, config=cfg)
        # File may or may not exist; the contract is "does not raise".

    def test_defensive_on_handler_failure(self, tmp_path, monkeypatch):
        cfg = _make_config(tmp_path)
        write_fallback_record({"a": 1}, config=cfg)

        original_emit = logging_handlers.RotatingFileHandler.emit

        def boom_emit(self, record):
            raise OSError("disk full")

        # Silence stdlib's stderr traceback during the failing-emit window.
        monkeypatch.setattr(logging.Handler, "handleError", lambda self, record: None)
        monkeypatch.setattr(logging_handlers.RotatingFileHandler, "emit", boom_emit)
        write_fallback_record({"b": 2}, config=cfg)

        # Restore emit and confirm subsequent writes still work.
        monkeypatch.setattr(logging_handlers.RotatingFileHandler, "emit", original_emit)
        write_fallback_record({"c": 3}, config=cfg)

        contents = cfg.fallback_path.read_text(encoding="utf-8")
        lines = [line for line in contents.splitlines() if line]
        payloads = [json.loads(line) for line in lines]
        keys_seen = {next(iter(p.keys())) for p in payloads}
        assert "a" in keys_seen
        assert "c" in keys_seen

    def test_routes_to_configured_path(self, tmp_path):
        cfg_a = ErrorLoggingConfig(
            fallback_path=tmp_path / "a.jsonl",
            fallback_max_bytes=5_000_000,
            fallback_backup_count=5,
        )
        cfg_b = ErrorLoggingConfig(
            fallback_path=tmp_path / "b.jsonl",
            fallback_max_bytes=5_000_000,
            fallback_backup_count=5,
        )
        write_fallback_record({"loc": "a"}, config=cfg_a)
        write_fallback_record({"loc": "b"}, config=cfg_b)
        assert json.loads(cfg_a.fallback_path.read_text("utf-8").strip()) == {"loc": "a"}
        assert json.loads(cfg_b.fallback_path.read_text("utf-8").strip()) == {"loc": "b"}

    def test_uses_streamlit_config_when_none_passed(self, tmp_path):
        custom = tmp_path / "from_streamlit.jsonl"
        custom_cfg = ErrorLoggingConfig(
            fallback_path=custom,
            fallback_max_bytes=5_000_000,
            fallback_backup_count=5,
        )
        with patch(
            "utils.error_fallback_sink.ErrorLoggingConfig.from_streamlit",
            return_value=custom_cfg,
        ):
            write_fallback_record({"src": "streamlit"})
        assert json.loads(custom.read_text("utf-8").strip()) == {"src": "streamlit"}


# ── read_fallback_records ─────────────────────────────────────────────────


class TestReadFallbackRecords:
    def _write(self, cfg: ErrorLoggingConfig, i: int, seconds: int) -> dict:
        payload = {
            "created_at": f"2026-06-05T12:00:{seconds:02d}",
            "i": i,
            "category": "sql_execution",
            "severity": "error",
            "error_type": "RuntimeError",
            "error_message": f"boom-{i}",
        }
        write_fallback_record(payload, config=cfg)
        return payload

    def test_read_empty_when_no_file(self, tmp_path):
        cfg = _make_config(tmp_path)
        records = read_fallback_records(since=datetime(2026, 1, 1), config=cfg)
        assert records == []

    def test_read_round_trips_written_records(self, tmp_path):
        cfg = _make_config(tmp_path)
        expected = [self._write(cfg, i, i) for i in range(3)]
        records = read_fallback_records(since=datetime(2026, 6, 5, 12, 0, 0), config=cfg)
        assert records == expected

    def test_read_filters_by_since(self, tmp_path):
        cfg = _make_config(tmp_path)
        for i in range(5):
            self._write(cfg, i, i)
        records = read_fallback_records(since=datetime(2026, 6, 5, 12, 0, 3), config=cfg)
        assert [r["i"] for r in records] == [3, 4]

    def test_read_filters_by_until(self, tmp_path):
        cfg = _make_config(tmp_path)
        for i in range(5):
            self._write(cfg, i, i)
        records = read_fallback_records(
            since=datetime(2026, 6, 5, 12, 0, 0),
            until=datetime(2026, 6, 5, 12, 0, 2),
            config=cfg,
        )
        assert [r["i"] for r in records] == [0, 1, 2]

    def test_read_respects_limit(self, tmp_path):
        cfg = _make_config(tmp_path)
        for i in range(5):
            self._write(cfg, i, i)
        records = read_fallback_records(since=datetime(2026, 6, 5, 12, 0, 0), limit=2, config=cfg)
        assert [r["i"] for r in records] == [0, 1]

    def test_read_returns_chronological_order(self, tmp_path):
        cfg = _make_config(tmp_path)
        # Write out of order
        self._write(cfg, 2, 20)
        self._write(cfg, 0, 0)
        self._write(cfg, 1, 10)
        records = read_fallback_records(since=datetime(2026, 6, 5, 12, 0, 0), config=cfg)
        assert [r["i"] for r in records] == [0, 1, 2]

    def test_read_includes_rotated_backups(self, tmp_path):
        cfg = ErrorLoggingConfig(
            fallback_path=tmp_path / "rot.jsonl",
            fallback_max_bytes=80,  # forces rotation after each ~42-byte record
            fallback_backup_count=5,
        )
        for i in range(4):
            self._write(cfg, i, i)
        # Confirm rotation actually happened
        rotated = [p.name for p in tmp_path.iterdir() if p.name.startswith("rot.jsonl")]
        assert any(name.endswith(".1") for name in rotated), f"expected rotation, got files: {rotated}"
        records = read_fallback_records(since=datetime(2026, 6, 5, 12, 0, 0), config=cfg)
        assert {r["i"] for r in records} == {0, 1, 2, 3}

    def test_read_skips_malformed_lines(self, tmp_path):
        cfg = _make_config(tmp_path)
        self._write(cfg, 1, 1)
        # Append garbage manually
        with cfg.fallback_path.open("a", encoding="utf-8") as f:
            f.write("not-json-at-all\n")
            f.write("{partial: json\n")
            f.write('"just-a-string"\n')  # valid JSON but not a dict
        self._write(cfg, 2, 2)
        records = read_fallback_records(since=datetime(2026, 6, 5, 12, 0, 0), config=cfg)
        assert [r["i"] for r in records] == [1, 2]

    def test_read_skips_records_without_created_at(self, tmp_path):
        cfg = _make_config(tmp_path)
        write_fallback_record({"i": 1}, config=cfg)  # no created_at
        self._write(cfg, 2, 5)
        records = read_fallback_records(since=datetime(2026, 6, 5, 12, 0, 0), config=cfg)
        assert [r["i"] for r in records] == [2]

    def test_read_skips_records_with_unparsable_created_at(self, tmp_path):
        cfg = _make_config(tmp_path)
        write_fallback_record({"created_at": "not-a-date", "i": 1}, config=cfg)
        self._write(cfg, 2, 5)
        records = read_fallback_records(since=datetime(2026, 6, 5, 12, 0, 0), config=cfg)
        assert [r["i"] for r in records] == [2]


# ── try_drain_fallback_to_db ──────────────────────────────────────────────


@pytest.fixture
def fresh_db(monkeypatch):
    from utils import error_fallback_sink

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    monkeypatch.setattr(error_fallback_sink, "SessionLocal", Session)
    return Session


class TestTryDrainFallbackToDb:
    def _write_full(self, cfg: ErrorLoggingConfig, i: int, seconds: int) -> dict:
        payload = {
            "created_at": f"2026-06-05T12:00:{seconds:02d}",
            "category": "sql_execution",
            "severity": "error",
            "error_type": "RuntimeError",
            "error_message": f"boom-{i}",
        }
        write_fallback_record(payload, config=cfg)
        return payload

    def test_drain_moves_records_to_db_and_clears_files(self, tmp_path, fresh_db):
        cfg = _make_config(tmp_path)
        for i in range(3):
            self._write_full(cfg, i, i)

        count = try_drain_fallback_to_db(config=cfg)

        assert count == 3
        assert not cfg.fallback_path.exists()
        with fresh_db() as session:
            rows = session.query(ErrorLog).order_by(ErrorLog.error_message).all()
            assert len(rows) == 3
            assert {r.error_message for r in rows} == {"boom-0", "boom-1", "boom-2"}
            assert all(r.category == "sql_execution" for r in rows)
            assert all(r.severity == "error" for r in rows)

    def test_drain_returns_zero_when_no_file(self, tmp_path, fresh_db):
        cfg = _make_config(tmp_path)
        count = try_drain_fallback_to_db(config=cfg)
        assert count == 0
        with fresh_db() as session:
            assert session.query(ErrorLog).count() == 0

    def test_drain_idempotent_on_existing_keys(self, tmp_path, fresh_db):
        cfg = _make_config(tmp_path)
        for i in range(2):
            self._write_full(cfg, i, i)

        first = try_drain_fallback_to_db(config=cfg)
        assert first == 2

        # Re-write the same logical records.
        for i in range(2):
            self._write_full(cfg, i, i)

        second = try_drain_fallback_to_db(config=cfg)
        assert second == 0
        with fresh_db() as session:
            assert session.query(ErrorLog).count() == 2

    def test_drain_skips_records_missing_required_fields(self, tmp_path, fresh_db):
        cfg = _make_config(tmp_path)
        self._write_full(cfg, 0, 0)
        # Missing error_type and error_message
        write_fallback_record(
            {
                "created_at": "2026-06-05T12:00:05",
                "category": "general",
                "severity": "error",
            },
            config=cfg,
        )

        count = try_drain_fallback_to_db(config=cfg)
        assert count == 1
        with fresh_db() as session:
            rows = session.query(ErrorLog).all()
            assert len(rows) == 1
            assert rows[0].error_message == "boom-0"

    def test_drain_handles_commit_failure(self, tmp_path, monkeypatch, fresh_db):
        cfg = _make_config(tmp_path)
        self._write_full(cfg, 0, 0)

        from utils import error_fallback_sink

        original_factory = error_fallback_sink.SessionLocal

        def broken_factory():
            session = original_factory()

            def boom():
                raise RuntimeError("commit failed")

            session.commit = boom
            return session

        monkeypatch.setattr(error_fallback_sink, "SessionLocal", broken_factory)

        count = try_drain_fallback_to_db(config=cfg)
        assert count == 0
        assert cfg.fallback_path.exists()

    def test_drain_does_not_raise_on_unexpected_errors(self, tmp_path, monkeypatch):
        cfg = _make_config(tmp_path)
        self._write_full(cfg, 0, 0)

        from utils import error_fallback_sink

        def broken_factory():
            raise RuntimeError("DB unreachable")

        monkeypatch.setattr(error_fallback_sink, "SessionLocal", broken_factory)

        count = try_drain_fallback_to_db(config=cfg)
        assert count == 0
        assert cfg.fallback_path.exists()

    def test_drain_preserves_field_round_trip(self, tmp_path, fresh_db):
        cfg = _make_config(tmp_path)
        payload = {
            "created_at": "2026-06-05T12:00:00",
            "category": "sql_execution",
            "severity": "critical",
            "error_type": "OperationalError",
            "error_message": "connection refused",
            "stack_trace": "Traceback (most recent call last):\n  File ...",
            "question": "How many patients?",
            "generated_sql": "SELECT COUNT(*) FROM patients",
            "llm_provider": "anthropic",
            "llm_model": "claude-3-5-sonnet",
            "context_data": '{"db_host": "localhost"}',
            "user_id": 42,
            "message_id": 100,
            "group_id": "uuid-abc-123",
            "auto_retry_attempted": True,
            "retry_successful": False,
            "retry_count": 3,
        }
        write_fallback_record(payload, config=cfg)

        count = try_drain_fallback_to_db(config=cfg)
        assert count == 1

        with fresh_db() as session:
            row = session.query(ErrorLog).first()
            assert row.category == "sql_execution"
            assert row.severity == "critical"
            assert row.error_type == "OperationalError"
            assert row.error_message == "connection refused"
            assert row.stack_trace.startswith("Traceback")
            assert row.question == "How many patients?"
            assert row.generated_sql == "SELECT COUNT(*) FROM patients"
            assert row.llm_provider == "anthropic"
            assert row.llm_model == "claude-3-5-sonnet"
            assert row.context_data == '{"db_host": "localhost"}'
            assert row.user_id == 42
            assert row.message_id == 100
            assert row.group_id == "uuid-abc-123"
            assert row.auto_retry_attempted is True
            assert row.retry_successful is False
            assert row.retry_count == 3
            assert row.created_at == datetime(2026, 6, 5, 12, 0, 0)


# ── log_error integration ─────────────────────────────────────────────────


class TestLogErrorIntegration:
    """Verify orm.logging_functions.log_error falls back to the JSONL sink
    when the SQLite write fails, and that the convenience wrappers inherit
    the same behavior because they delegate to log_error."""

    def _patch_streamlit_config(self, monkeypatch, cfg: ErrorLoggingConfig):
        monkeypatch.setattr(
            "utils.error_fallback_sink.ErrorLoggingConfig.from_streamlit",
            classmethod(lambda _cls: cfg),
        )

    def _patch_session_factory_to_raise(self, monkeypatch):
        from orm import logging_functions

        def boom_factory():
            raise RuntimeError("DB unreachable")

        monkeypatch.setattr(logging_functions, "SessionLocal", boom_factory)

    def test_log_error_falls_back_when_session_raises(self, tmp_path, monkeypatch):
        cfg = _make_config(tmp_path)
        self._patch_streamlit_config(monkeypatch, cfg)
        self._patch_session_factory_to_raise(monkeypatch)

        from orm import logging_functions

        result = logging_functions.log_error(
            category=ErrorCategory.SQL_EXECUTION,
            severity=ErrorSeverity.ERROR,
            error_type="OperationalError",
            error_message="connection refused",
            user_id=42,
            group_id="grp-xyz",
            include_traceback=False,
        )

        assert result is None
        records = read_fallback_records(since=datetime(2000, 1, 1), config=cfg)
        assert len(records) == 1
        rec = records[0]
        assert rec["category"] == "sql_execution"
        assert rec["severity"] == "error"
        assert rec["error_type"] == "OperationalError"
        assert rec["error_message"] == "connection refused"
        assert rec["user_id"] == 42
        assert rec["group_id"] == "grp-xyz"
        assert "created_at" in rec

    def test_log_sql_generation_error_inherits_fallback(self, tmp_path, monkeypatch):
        cfg = _make_config(tmp_path)
        self._patch_streamlit_config(monkeypatch, cfg)
        self._patch_session_factory_to_raise(monkeypatch)

        from orm import logging_functions

        result = logging_functions.log_sql_generation_error(
            error=ValueError("bad sql"),
            question="how many patients?",
            user_id=1,
            llm_provider="anthropic",
            llm_model="claude-3-5-sonnet",
            group_id="grp-abc",
        )

        assert result is None
        records = read_fallback_records(since=datetime(2000, 1, 1), config=cfg)
        assert len(records) == 1
        rec = records[0]
        assert rec["category"] == "sql_generation"
        assert rec["error_type"] == "ValueError"
        assert rec["error_message"] == "bad sql"
        assert rec["question"] == "how many patients?"
        assert rec["llm_provider"] == "anthropic"
        assert rec["group_id"] == "grp-abc"

    def test_log_sql_execution_error_inherits_fallback(self, tmp_path, monkeypatch):
        cfg = _make_config(tmp_path)
        self._patch_streamlit_config(monkeypatch, cfg)
        self._patch_session_factory_to_raise(monkeypatch)

        from orm import logging_functions

        result = logging_functions.log_sql_execution_error(
            error=RuntimeError("syntax error near WHERE"),
            sql="SELECT bad FROM nope",
            question="bad query",
            user_id=1,
        )

        assert result is None
        records = read_fallback_records(since=datetime(2000, 1, 1), config=cfg)
        assert len(records) == 1
        rec = records[0]
        assert rec["category"] == "sql_execution"
        assert rec["error_type"] == "RuntimeError"
        assert rec["generated_sql"] == "SELECT bad FROM nope"

    def test_log_chart_generation_error_inherits_fallback(self, tmp_path, monkeypatch):
        cfg = _make_config(tmp_path)
        self._patch_streamlit_config(monkeypatch, cfg)
        self._patch_session_factory_to_raise(monkeypatch)

        from orm import logging_functions

        result = logging_functions.log_chart_generation_error(
            error=TypeError("not a dataframe"),
            user_id=1,
            question="show chart",
        )

        assert result is None
        records = read_fallback_records(since=datetime(2000, 1, 1), config=cfg)
        assert len(records) == 1
        assert records[0]["category"] == "chart_generation"
        assert records[0]["error_type"] == "TypeError"

    def test_log_error_does_not_raise_when_both_db_and_fallback_fail(self, tmp_path, monkeypatch):
        cfg = _make_config(tmp_path)
        self._patch_streamlit_config(monkeypatch, cfg)
        self._patch_session_factory_to_raise(monkeypatch)

        # Make write_fallback_record itself raise (it normally swallows; we
        # force a raise to verify log_error's own outer guard).
        def boom_write(*args, **kwargs):
            raise RuntimeError("fallback also broken")

        monkeypatch.setattr(
            "orm.logging_functions.write_fallback_record",
            boom_write,
        )

        from orm import logging_functions

        result = logging_functions.log_error(
            category=ErrorCategory.GENERAL,
            severity=ErrorSeverity.ERROR,
            error_type="X",
            error_message="y",
            include_traceback=False,
        )
        assert result is None

    def test_log_error_does_not_write_fallback_on_db_success(self, tmp_path, monkeypatch, fresh_db):
        cfg = _make_config(tmp_path)
        self._patch_streamlit_config(monkeypatch, cfg)

        # Also point orm.logging_functions.SessionLocal at the same in-memory engine
        # so the DB write succeeds via the fresh_db fixture's session.
        from orm import logging_functions
        from utils import error_fallback_sink

        monkeypatch.setattr(logging_functions, "SessionLocal", error_fallback_sink.SessionLocal)

        result = logging_functions.log_error(
            category=ErrorCategory.SQL_EXECUTION,
            severity=ErrorSeverity.ERROR,
            error_type="OK",
            error_message="happy path",
            include_traceback=False,
        )

        assert result is not None
        assert result.error_message == "happy path"
        # Fallback file must not have been touched.
        assert not cfg.fallback_path.exists()

    def test_log_error_fallback_preserves_caller_traceback(self, tmp_path, monkeypatch):
        cfg = _make_config(tmp_path)
        self._patch_streamlit_config(monkeypatch, cfg)
        self._patch_session_factory_to_raise(monkeypatch)

        from orm import logging_functions

        try:
            raise ValueError("caller-original-error")
        except ValueError:
            result = logging_functions.log_error(
                category=ErrorCategory.GENERAL,
                severity=ErrorSeverity.ERROR,
                error_type="ValueError",
                error_message="caller-original-error",
                include_traceback=True,
            )

        assert result is None
        records = read_fallback_records(since=datetime(2000, 1, 1), config=cfg)
        assert len(records) == 1
        stack_trace = records[0].get("stack_trace")
        assert stack_trace is not None
        assert "caller-original-error" in stack_trace
