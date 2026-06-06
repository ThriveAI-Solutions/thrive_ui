"""Tests for orm.error_read_model (Phase 1: types + to_dict)."""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from orm import error_read_model
from orm.error_read_model import (
    DEFAULT_SOURCES,
    ErrorRow,
    ErrorSource,
    _read_from_agent_run,
    _read_from_error_log,
    _read_from_fallback,
    count_errors_by_source,
    query_errors,
)
from orm.models import AgentRun, Base, ErrorLog
from utils.error_fallback_sink import (
    ErrorLoggingConfig,
    _reset_handler_cache,
    write_fallback_record,
)


@pytest.fixture(autouse=True)
def _reset_fallback_handler_cache():
    _reset_handler_cache()
    yield
    _reset_handler_cache()


@pytest.fixture
def fresh_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    return Session


def _seed_error_log(
    session,
    **overrides,
) -> ErrorLog:
    """Insert one ErrorLog row and return the persisted instance."""
    defaults = dict(
        category="sql_execution",
        severity="error",
        error_type="RuntimeError",
        error_message="boom",
        user_id=1,
        created_at=datetime(2026, 6, 5, 12, 0, 0),
    )
    defaults.update(overrides)
    row = ErrorLog(**defaults)
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


_RUN_COUNTER = [0]


def _seed_agent_run(
    session,
    **overrides,
) -> AgentRun:
    """Insert one AgentRun row with sensible failure defaults."""
    _RUN_COUNTER[0] += 1
    defaults = dict(
        run_id=f"run-{_RUN_COUNTER[0]}",
        session_id="sess-1",
        user_id=1,
        user_role=0,
        status="error",
        success=False,
        error_type="RuntimeError",
        error="agent boom",
        created_at=datetime(2026, 6, 5, 12, 0, 0),
    )
    defaults.update(overrides)
    row = AgentRun(**defaults)
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


def _minimal_row(**overrides) -> ErrorRow:
    base = dict(
        id="error_log:1",
        source=ErrorSource.ERROR_LOG,
        created_at=datetime(2026, 6, 5, 12, 0, 0),
        user_id=None,
        category=None,
        severity=None,
        error_type=None,
        error_message=None,
        stack_trace=None,
        question=None,
        generated_sql=None,
        llm_provider=None,
        llm_model=None,
        context_data=None,
        group_id=None,
        run_id=None,
        message_id=None,
        auto_retry_attempted=None,
        retry_successful=None,
        retry_count=None,
    )
    base.update(overrides)
    return ErrorRow(**base)


# ── ErrorSource enum ──────────────────────────────────────────────────────


class TestErrorSourceEnum:
    def test_values(self):
        assert ErrorSource.ERROR_LOG.value == "error_log"
        assert ErrorSource.AGENT_RUN.value == "agent_run"
        assert ErrorSource.FALLBACK_SINK.value == "fallback_sink"

    def test_membership(self):
        assert {s.value for s in ErrorSource} == {"error_log", "agent_run", "fallback_sink"}


# ── ErrorRow dataclass ────────────────────────────────────────────────────


class TestErrorRow:
    def test_minimum_fields_with_nones(self):
        row = _minimal_row()
        assert row.id == "error_log:1"
        assert row.source is ErrorSource.ERROR_LOG
        assert row.created_at == datetime(2026, 6, 5, 12, 0, 0)
        assert row.error_message is None

    def test_full_fields_round_trip(self):
        row = _minimal_row(
            id="agent_run:r-1",
            source=ErrorSource.AGENT_RUN,
            user_id=7,
            category="agent_run",
            severity="warning",
            error_type="ToolCapReached",
            error_message="too many tool calls",
            stack_trace="Traceback ...",
            question="show me the cohort",
            generated_sql="SELECT 1",
            llm_provider="anthropic",
            llm_model="claude-opus",
            context_data='{"k":"v"}',
            group_id="grp-1",
            run_id="r-1",
            message_id=42,
            auto_retry_attempted=True,
            retry_successful=False,
            retry_count=3,
        )
        assert row.user_id == 7
        assert row.category == "agent_run"
        assert row.severity == "warning"
        assert row.run_id == "r-1"
        assert row.auto_retry_attempted is True
        assert row.retry_successful is False
        assert row.retry_count == 3

    def test_is_frozen(self):
        row = _minimal_row()
        with pytest.raises(dataclasses.FrozenInstanceError):
            row.error_message = "mutating frozen dataclass should raise"  # type: ignore[misc]

    def test_equality(self):
        a = _minimal_row(error_message="boom")
        b = _minimal_row(error_message="boom")
        c = _minimal_row(error_message="different")
        assert a == b
        assert a != c
        assert hash(a) == hash(b)


# ── ErrorRow.to_dict ──────────────────────────────────────────────────────


class TestToDict:
    def test_returns_plain_dict(self):
        d = _minimal_row().to_dict()
        assert isinstance(d, dict)
        assert not dataclasses.is_dataclass(d)

    def test_enum_source_becomes_string_value(self):
        d = _minimal_row(source=ErrorSource.AGENT_RUN).to_dict()
        assert d["source"] == "agent_run"
        assert not isinstance(d["source"], ErrorSource)

    def test_datetime_created_at_becomes_iso_string(self):
        when = datetime(2026, 6, 5, 12, 34, 56)
        d = _minimal_row(created_at=when).to_dict()
        assert d["created_at"] == when.isoformat()
        assert isinstance(d["created_at"], str)

    def test_preserves_none_for_optional_fields(self):
        d = _minimal_row().to_dict()
        assert d["error_message"] is None
        assert d["stack_trace"] is None
        assert d["run_id"] is None
        assert d["context_data"] is None

    def test_is_json_serializable_minimal(self):
        payload = _minimal_row().to_dict()
        # Must not raise — proves end-to-end serializability.
        json.dumps(payload)

    def test_is_json_serializable_full(self):
        row = _minimal_row(
            id="fallback_sink:2026-06-05T12:00:00:RuntimeError:1234",
            source=ErrorSource.FALLBACK_SINK,
            user_id=99,
            category="sql_execution",
            severity="error",
            error_type="RuntimeError",
            error_message="boom",
            stack_trace="Traceback...",
            question="how many?",
            generated_sql="SELECT COUNT(*) FROM patients",
            llm_provider="anthropic",
            llm_model="claude",
            context_data='{"db_host":"localhost"}',
            group_id="grp-1",
            run_id=None,
            message_id=100,
            auto_retry_attempted=False,
            retry_successful=None,
            retry_count=0,
        )
        payload = row.to_dict()
        json.dumps(payload)
        assert payload["source"] == "fallback_sink"
        assert payload["retry_count"] == 0
        assert payload["retry_successful"] is None

    def test_keys_match_dataclass_fields(self):
        d = _minimal_row().to_dict()
        expected = {f.name for f in dataclasses.fields(ErrorRow)}
        assert set(d.keys()) == expected


# ── _read_from_error_log adapter ──────────────────────────────────────────


class TestReadFromErrorLog:
    EPOCH = datetime(2000, 1, 1)

    def test_returns_empty_when_no_rows(self, fresh_db):
        with fresh_db() as session:
            result = _read_from_error_log(session, since=self.EPOCH)
        assert result == []

    def test_returns_rows_in_time_range(self, fresh_db):
        with fresh_db() as session:
            _seed_error_log(session, error_message="a")
            _seed_error_log(session, error_message="b")
            result = _read_from_error_log(session, since=self.EPOCH)
        assert {r.error_message for r in result} == {"a", "b"}
        assert all(r.source is ErrorSource.ERROR_LOG for r in result)

    def test_excludes_rows_before_since(self, fresh_db):
        with fresh_db() as session:
            _seed_error_log(
                session,
                error_message="old",
                created_at=datetime(2025, 1, 1),
            )
            _seed_error_log(
                session,
                error_message="new",
                created_at=datetime(2026, 6, 5),
            )
            result = _read_from_error_log(
                session,
                since=datetime(2026, 1, 1),
            )
        assert [r.error_message for r in result] == ["new"]

    def test_excludes_rows_after_until(self, fresh_db):
        with fresh_db() as session:
            _seed_error_log(
                session,
                error_message="in",
                created_at=datetime(2026, 6, 5),
            )
            _seed_error_log(
                session,
                error_message="future",
                created_at=datetime(2026, 12, 1),
            )
            result = _read_from_error_log(
                session,
                since=self.EPOCH,
                until=datetime(2026, 11, 1),
            )
        assert [r.error_message for r in result] == ["in"]

    def test_open_ended_when_until_is_none(self, fresh_db):
        with fresh_db() as session:
            _seed_error_log(session, error_message="a")
            _seed_error_log(session, error_message="b")
            result = _read_from_error_log(session, since=self.EPOCH, until=None)
        assert len(result) == 2

    def test_filters_by_categories_in(self, fresh_db):
        with fresh_db() as session:
            _seed_error_log(session, category="sql_execution", error_message="x")
            _seed_error_log(session, category="sql_generation", error_message="y")
            _seed_error_log(session, category="chart_generation", error_message="z")
            result = _read_from_error_log(
                session,
                since=self.EPOCH,
                categories={"sql_execution", "chart_generation"},
            )
        assert {r.error_message for r in result} == {"x", "z"}

    def test_filters_by_severities_in(self, fresh_db):
        with fresh_db() as session:
            _seed_error_log(session, severity="error", error_message="x")
            _seed_error_log(session, severity="warning", error_message="y")
            _seed_error_log(session, severity="critical", error_message="z")
            result = _read_from_error_log(
                session,
                since=self.EPOCH,
                severities={"warning", "critical"},
            )
        assert {r.error_message for r in result} == {"y", "z"}

    def test_filters_by_user_id(self, fresh_db):
        with fresh_db() as session:
            _seed_error_log(session, user_id=1, error_message="me")
            _seed_error_log(session, user_id=2, error_message="someone-else")
            result = _read_from_error_log(session, since=self.EPOCH, user_id=1)
        assert [r.error_message for r in result] == ["me"]

    def test_search_matches_error_message(self, fresh_db):
        with fresh_db() as session:
            _seed_error_log(session, error_message="connection refused")
            _seed_error_log(session, error_message="timeout")
            result = _read_from_error_log(session, since=self.EPOCH, search="refused")
        assert [r.error_message for r in result] == ["connection refused"]

    def test_search_matches_question(self, fresh_db):
        with fresh_db() as session:
            _seed_error_log(session, error_message="x", question="how many patients?")
            _seed_error_log(session, error_message="y", question="show charts")
            result = _read_from_error_log(session, since=self.EPOCH, search="patients")
        assert [r.error_message for r in result] == ["x"]

    def test_search_is_case_insensitive(self, fresh_db):
        with fresh_db() as session:
            _seed_error_log(session, error_message="Connection Refused")
            result = _read_from_error_log(session, since=self.EPOCH, search="connection")
        assert len(result) == 1

    def test_multiple_filters_combine_with_and(self, fresh_db):
        with fresh_db() as session:
            # Two rows with same category but different severity / user
            _seed_error_log(
                session,
                category="sql_execution",
                severity="error",
                user_id=1,
                error_message="match",
            )
            _seed_error_log(
                session,
                category="sql_execution",
                severity="warning",
                user_id=1,
                error_message="wrong-sev",
            )
            _seed_error_log(
                session,
                category="sql_execution",
                severity="error",
                user_id=2,
                error_message="wrong-user",
            )
            result = _read_from_error_log(
                session,
                since=self.EPOCH,
                categories={"sql_execution"},
                severities={"error"},
                user_id=1,
            )
        assert [r.error_message for r in result] == ["match"]

    def test_maps_all_fields_to_error_row(self, fresh_db):
        with fresh_db() as session:
            persisted = _seed_error_log(
                session,
                category="sql_execution",
                severity="critical",
                error_type="OperationalError",
                error_message="connection refused",
                stack_trace="Traceback...",
                question="how many?",
                generated_sql="SELECT 1",
                llm_provider="anthropic",
                llm_model="claude",
                context_data='{"k":"v"}',
                group_id="grp-1",
                message_id=99,
                user_id=42,
                auto_retry_attempted=True,
                retry_successful=False,
                retry_count=3,
            )
            result = _read_from_error_log(session, since=self.EPOCH)
        assert len(result) == 1
        row = result[0]
        assert row.id == f"error_log:{persisted.id}"
        assert row.source is ErrorSource.ERROR_LOG
        assert row.category == "sql_execution"
        assert row.severity == "critical"
        assert row.error_type == "OperationalError"
        assert row.error_message == "connection refused"
        assert row.stack_trace == "Traceback..."
        assert row.question == "how many?"
        assert row.generated_sql == "SELECT 1"
        assert row.llm_provider == "anthropic"
        assert row.llm_model == "claude"
        assert row.context_data == '{"k":"v"}'
        assert row.group_id == "grp-1"
        assert row.message_id == 99
        assert row.user_id == 42
        assert row.auto_retry_attempted is True
        assert row.retry_successful is False
        assert row.retry_count == 3
        assert row.run_id is None  # ErrorLog has no run_id column

    def test_id_is_source_prefixed(self, fresh_db):
        with fresh_db() as session:
            persisted = _seed_error_log(session)
            result = _read_from_error_log(session, since=self.EPOCH)
        assert result[0].id == f"error_log:{persisted.id}"

    def test_source_is_error_log_enum(self, fresh_db):
        with fresh_db() as session:
            _seed_error_log(session)
            result = _read_from_error_log(session, since=self.EPOCH)
        assert result[0].source is ErrorSource.ERROR_LOG


# ── _read_from_agent_run adapter ──────────────────────────────────────────


class TestReadFromAgentRun:
    EPOCH = datetime(2000, 1, 1)

    def test_returns_empty_when_no_rows(self, fresh_db):
        with fresh_db() as session:
            result = _read_from_agent_run(session, since=self.EPOCH)
        assert result == []

    def test_excludes_successful_runs(self, fresh_db):
        with fresh_db() as session:
            _seed_agent_run(session, success=True, error_type=None, error=None)
            _seed_agent_run(session, success=False, error="boom")
            result = _read_from_agent_run(session, since=self.EPOCH)
        assert [r.error_message for r in result] == ["boom"]

    def test_includes_runs_with_error_type_even_if_success(self, fresh_db):
        """Defensive: if a row somehow has both success=True AND a populated
        error_type (e.g. a partial failure that ended OK), surface it."""
        with fresh_db() as session:
            _seed_agent_run(
                session,
                success=True,
                error_type="WeirdState",
                error="something off",
            )
            result = _read_from_agent_run(session, since=self.EPOCH)
        assert len(result) == 1
        assert result[0].error_type == "WeirdState"

    def test_category_is_constant_agent_run(self, fresh_db):
        with fresh_db() as session:
            _seed_agent_run(session)
            result = _read_from_agent_run(session, since=self.EPOCH)
        assert result[0].category == "agent_run"

    def test_severity_is_warning_when_status_cap_reached(self, fresh_db):
        with fresh_db() as session:
            _seed_agent_run(session, status="cap_reached")
            result = _read_from_agent_run(session, since=self.EPOCH)
        assert result[0].severity == "warning"

    def test_severity_is_error_otherwise(self, fresh_db):
        with fresh_db() as session:
            _seed_agent_run(session, status="error")
            result = _read_from_agent_run(session, since=self.EPOCH)
        assert result[0].severity == "error"

    def test_id_and_run_id_are_populated(self, fresh_db):
        with fresh_db() as session:
            row = _seed_agent_run(session, run_id="my-run-id")
            result = _read_from_agent_run(session, since=self.EPOCH)
        assert result[0].id == "agent_run:my-run-id"
        assert result[0].run_id == "my-run-id"
        # Sanity: persisted row matches what we expect
        assert row.run_id == "my-run-id"

    def test_source_is_agent_run_enum(self, fresh_db):
        with fresh_db() as session:
            _seed_agent_run(session)
            result = _read_from_agent_run(session, since=self.EPOCH)
        assert result[0].source is ErrorSource.AGENT_RUN

    def test_error_message_comes_from_error_column(self, fresh_db):
        with fresh_db() as session:
            _seed_agent_run(session, error="agent failed at step 3")
            result = _read_from_agent_run(session, since=self.EPOCH)
        assert result[0].error_message == "agent failed at step 3"

    def test_maps_supporting_fields(self, fresh_db):
        with fresh_db() as session:
            _seed_agent_run(
                session,
                stack_trace="Traceback ...",
                question="show me cohort",
                llm_provider="anthropic",
                llm_model="claude-opus",
                group_id="grp-1",
                final_message_id=42,
                user_id=7,
            )
            result = _read_from_agent_run(session, since=self.EPOCH)
        row = result[0]
        assert row.stack_trace == "Traceback ..."
        assert row.question == "show me cohort"
        assert row.llm_provider == "anthropic"
        assert row.llm_model == "claude-opus"
        assert row.group_id == "grp-1"
        assert row.message_id == 42
        assert row.user_id == 7
        # AgentRun has no per-run generated_sql / context_data / retry fields
        assert row.generated_sql is None
        assert row.context_data is None
        assert row.auto_retry_attempted is None
        assert row.retry_successful is None
        assert row.retry_count is None

    def test_filter_by_since(self, fresh_db):
        with fresh_db() as session:
            _seed_agent_run(session, error="old", created_at=datetime(2025, 1, 1))
            _seed_agent_run(session, error="new", created_at=datetime(2026, 6, 5))
            result = _read_from_agent_run(session, since=datetime(2026, 1, 1))
        assert [r.error_message for r in result] == ["new"]

    def test_filter_by_until(self, fresh_db):
        with fresh_db() as session:
            _seed_agent_run(session, error="in", created_at=datetime(2026, 6, 5))
            _seed_agent_run(session, error="future", created_at=datetime(2026, 12, 1))
            result = _read_from_agent_run(
                session,
                since=self.EPOCH,
                until=datetime(2026, 11, 1),
            )
        assert [r.error_message for r in result] == ["in"]

    def test_filter_by_user_id(self, fresh_db):
        with fresh_db() as session:
            _seed_agent_run(session, user_id=1, error="me")
            _seed_agent_run(session, user_id=2, error="other")
            result = _read_from_agent_run(session, since=self.EPOCH, user_id=1)
        assert [r.error_message for r in result] == ["me"]

    def test_categories_filter_excludes_when_agent_run_not_in_set(self, fresh_db):
        """Since the synthesized category is the constant 'agent_run', any
        categories filter that doesn't include it should return zero rows."""
        with fresh_db() as session:
            _seed_agent_run(session)
            result = _read_from_agent_run(
                session,
                since=self.EPOCH,
                categories={"sql_execution"},
            )
        assert result == []

    def test_categories_filter_includes_when_agent_run_in_set(self, fresh_db):
        with fresh_db() as session:
            _seed_agent_run(session, error="kept")
            result = _read_from_agent_run(
                session,
                since=self.EPOCH,
                categories={"sql_execution", "agent_run"},
            )
        assert [r.error_message for r in result] == ["kept"]

    def test_severities_filter(self, fresh_db):
        with fresh_db() as session:
            _seed_agent_run(session, status="cap_reached", error="cap-row")
            _seed_agent_run(session, status="error", error="err-row")
            result = _read_from_agent_run(
                session,
                since=self.EPOCH,
                severities={"warning"},
            )
        assert [r.error_message for r in result] == ["cap-row"]

    def test_search_matches_error(self, fresh_db):
        with fresh_db() as session:
            _seed_agent_run(session, error="connection refused")
            _seed_agent_run(session, error="timeout")
            result = _read_from_agent_run(session, since=self.EPOCH, search="refused")
        assert [r.error_message for r in result] == ["connection refused"]

    def test_search_matches_question(self, fresh_db):
        with fresh_db() as session:
            _seed_agent_run(session, error="x", question="show me patients")
            _seed_agent_run(session, error="y", question="something else")
            result = _read_from_agent_run(session, since=self.EPOCH, search="patients")
        assert [r.error_message for r in result] == ["x"]


# ── _read_from_fallback adapter ───────────────────────────────────────────


def _make_fallback_cfg(tmp_path) -> ErrorLoggingConfig:
    return ErrorLoggingConfig(
        fallback_path=tmp_path / "fallback.jsonl",
        fallback_max_bytes=5_000_000,
        fallback_backup_count=5,
    )


def _seed_fallback(cfg: ErrorLoggingConfig, *, seconds: int = 0, **overrides) -> dict:
    payload = dict(
        created_at=f"2026-06-05T12:00:{seconds:02d}",
        category="sql_execution",
        severity="error",
        error_type="RuntimeError",
        error_message="fallback boom",
        user_id=1,
    )
    payload.update(overrides)
    write_fallback_record(payload, config=cfg)
    return payload


class TestReadFromFallback:
    EPOCH = datetime(2000, 1, 1)

    def test_returns_empty_when_no_file(self, tmp_path):
        cfg = _make_fallback_cfg(tmp_path)
        result = _read_from_fallback(self.EPOCH, fallback_config=cfg)
        assert result == []

    def test_returns_records_mapped_to_error_rows(self, tmp_path):
        cfg = _make_fallback_cfg(tmp_path)
        _seed_fallback(cfg, error_message="a")
        _seed_fallback(cfg, seconds=1, error_message="b")
        result = _read_from_fallback(self.EPOCH, fallback_config=cfg)
        assert {r.error_message for r in result} == {"a", "b"}
        assert all(isinstance(r, ErrorRow) for r in result)

    def test_source_is_fallback_sink_enum(self, tmp_path):
        cfg = _make_fallback_cfg(tmp_path)
        _seed_fallback(cfg)
        result = _read_from_fallback(self.EPOCH, fallback_config=cfg)
        assert result[0].source is ErrorSource.FALLBACK_SINK

    def test_id_is_source_prefixed(self, tmp_path):
        cfg = _make_fallback_cfg(tmp_path)
        _seed_fallback(cfg, error_type="RuntimeError", error_message="x")
        result = _read_from_fallback(self.EPOCH, fallback_config=cfg)
        assert result[0].id.startswith("fallback_sink:")
        # Should include created_at + error_type in the synthesized id
        assert "RuntimeError" in result[0].id

    def test_run_id_is_none(self, tmp_path):
        cfg = _make_fallback_cfg(tmp_path)
        _seed_fallback(cfg)
        result = _read_from_fallback(self.EPOCH, fallback_config=cfg)
        assert result[0].run_id is None

    def test_filter_by_categories(self, tmp_path):
        cfg = _make_fallback_cfg(tmp_path)
        _seed_fallback(cfg, category="sql_execution", error_message="x")
        _seed_fallback(cfg, seconds=1, category="chart_generation", error_message="y")
        _seed_fallback(cfg, seconds=2, category="sql_generation", error_message="z")
        result = _read_from_fallback(
            self.EPOCH,
            categories={"sql_execution", "chart_generation"},
            fallback_config=cfg,
        )
        assert {r.error_message for r in result} == {"x", "y"}

    def test_filter_by_severities(self, tmp_path):
        cfg = _make_fallback_cfg(tmp_path)
        _seed_fallback(cfg, severity="error", error_message="x")
        _seed_fallback(cfg, seconds=1, severity="warning", error_message="y")
        _seed_fallback(cfg, seconds=2, severity="critical", error_message="z")
        result = _read_from_fallback(
            self.EPOCH,
            severities={"warning", "critical"},
            fallback_config=cfg,
        )
        assert {r.error_message for r in result} == {"y", "z"}

    def test_filter_by_user_id(self, tmp_path):
        cfg = _make_fallback_cfg(tmp_path)
        _seed_fallback(cfg, user_id=1, error_message="me")
        _seed_fallback(cfg, seconds=1, user_id=2, error_message="other")
        result = _read_from_fallback(self.EPOCH, user_id=1, fallback_config=cfg)
        assert [r.error_message for r in result] == ["me"]

    def test_search_matches_error_message_case_insensitive(self, tmp_path):
        cfg = _make_fallback_cfg(tmp_path)
        _seed_fallback(cfg, error_message="Connection Refused")
        _seed_fallback(cfg, seconds=1, error_message="Timeout")
        result = _read_from_fallback(self.EPOCH, search="connection", fallback_config=cfg)
        assert [r.error_message for r in result] == ["Connection Refused"]

    def test_search_matches_question_case_insensitive(self, tmp_path):
        cfg = _make_fallback_cfg(tmp_path)
        _seed_fallback(cfg, error_message="x", question="How many Patients?")
        _seed_fallback(cfg, seconds=1, error_message="y", question="Show charts")
        result = _read_from_fallback(self.EPOCH, search="patients", fallback_config=cfg)
        assert [r.error_message for r in result] == ["x"]

    def test_filter_by_until(self, tmp_path):
        cfg = _make_fallback_cfg(tmp_path)
        _seed_fallback(cfg, created_at="2026-06-05T12:00:00", error_message="in")
        _seed_fallback(cfg, created_at="2026-12-01T12:00:00", error_message="future")
        result = _read_from_fallback(
            self.EPOCH,
            until=datetime(2026, 11, 1),
            fallback_config=cfg,
        )
        assert [r.error_message for r in result] == ["in"]

    def test_maps_all_fields(self, tmp_path):
        cfg = _make_fallback_cfg(tmp_path)
        _seed_fallback(
            cfg,
            category="sql_execution",
            severity="critical",
            error_type="OperationalError",
            error_message="connection refused",
            stack_trace="Traceback...",
            question="how many?",
            generated_sql="SELECT 1",
            llm_provider="anthropic",
            llm_model="claude",
            context_data='{"k":"v"}',
            group_id="grp-1",
            message_id=99,
            user_id=42,
            auto_retry_attempted=True,
            retry_successful=False,
            retry_count=3,
        )
        result = _read_from_fallback(self.EPOCH, fallback_config=cfg)
        assert len(result) == 1
        row = result[0]
        assert row.category == "sql_execution"
        assert row.severity == "critical"
        assert row.error_type == "OperationalError"
        assert row.error_message == "connection refused"
        assert row.stack_trace == "Traceback..."
        assert row.question == "how many?"
        assert row.generated_sql == "SELECT 1"
        assert row.llm_provider == "anthropic"
        assert row.llm_model == "claude"
        assert row.context_data == '{"k":"v"}'
        assert row.group_id == "grp-1"
        assert row.message_id == 99
        assert row.user_id == 42
        assert row.auto_retry_attempted is True
        assert row.retry_successful is False
        assert row.retry_count == 3
        assert row.run_id is None  # fallback records have no run_id

    def test_combines_filters_with_and(self, tmp_path):
        cfg = _make_fallback_cfg(tmp_path)
        _seed_fallback(
            cfg,
            category="sql_execution",
            severity="error",
            user_id=1,
            error_message="match",
        )
        _seed_fallback(
            cfg,
            seconds=1,
            category="sql_execution",
            severity="warning",
            user_id=1,
            error_message="wrong-sev",
        )
        _seed_fallback(
            cfg,
            seconds=2,
            category="sql_execution",
            severity="error",
            user_id=2,
            error_message="wrong-user",
        )
        result = _read_from_fallback(
            self.EPOCH,
            categories={"sql_execution"},
            severities={"error"},
            user_id=1,
            fallback_config=cfg,
        )
        assert [r.error_message for r in result] == ["match"]


# ── query_errors public entry ─────────────────────────────────────────────


@pytest.fixture
def patched_session_local(fresh_db, monkeypatch):
    """Bind error_read_model.SessionLocal to an in-memory engine."""
    Session = fresh_db
    monkeypatch.setattr(error_read_model, "SessionLocal", Session)
    return Session


class TestQueryErrors:
    EPOCH = datetime(2000, 1, 1)

    def test_default_sources_constant_includes_all_three(self):
        assert DEFAULT_SOURCES == {
            ErrorSource.ERROR_LOG,
            ErrorSource.AGENT_RUN,
            ErrorSource.FALLBACK_SINK,
        }

    def test_default_sources_merges_all_three(self, patched_session_local, tmp_path):
        Session = patched_session_local
        cfg = _make_fallback_cfg(tmp_path)
        with Session() as session:
            _seed_error_log(
                session,
                error_message="from-error-log",
                created_at=datetime(2026, 6, 5, 12, 0, 0),
            )
            _seed_agent_run(
                session,
                error="from-agent-run",
                created_at=datetime(2026, 6, 5, 12, 0, 1),
            )
        _seed_fallback(
            cfg,
            error_message="from-fallback",
            created_at="2026-06-05T12:00:02",
        )

        result = query_errors(self.EPOCH, fallback_config=cfg)

        sources = {r.source for r in result}
        assert sources == {
            ErrorSource.ERROR_LOG,
            ErrorSource.AGENT_RUN,
            ErrorSource.FALLBACK_SINK,
        }
        messages = {r.error_message for r in result}
        assert messages == {"from-error-log", "from-agent-run", "from-fallback"}

    def test_sources_subset_returns_only_those(self, patched_session_local, tmp_path):
        Session = patched_session_local
        cfg = _make_fallback_cfg(tmp_path)
        with Session() as session:
            _seed_error_log(session, error_message="db")
            _seed_agent_run(session, error="agent")
        _seed_fallback(cfg, error_message="fb")

        result = query_errors(
            self.EPOCH,
            sources={ErrorSource.ERROR_LOG},
            fallback_config=cfg,
        )
        assert {r.source for r in result} == {ErrorSource.ERROR_LOG}
        assert {r.error_message for r in result} == {"db"}

    def test_empty_sources_returns_empty(self, patched_session_local, tmp_path):
        Session = patched_session_local
        cfg = _make_fallback_cfg(tmp_path)
        with Session() as session:
            _seed_error_log(session)
        _seed_fallback(cfg)

        result = query_errors(self.EPOCH, sources=set(), fallback_config=cfg)
        assert result == []

    def test_sorts_by_created_at_descending(self, patched_session_local, tmp_path):
        Session = patched_session_local
        cfg = _make_fallback_cfg(tmp_path)
        with Session() as session:
            _seed_error_log(
                session,
                error_message="middle",
                created_at=datetime(2026, 6, 5, 12, 0, 1),
            )
            _seed_error_log(
                session,
                error_message="oldest",
                created_at=datetime(2026, 6, 5, 12, 0, 0),
            )
        _seed_fallback(
            cfg,
            error_message="newest",
            created_at="2026-06-05T12:00:02",
        )

        result = query_errors(self.EPOCH, fallback_config=cfg)
        assert [r.error_message for r in result] == ["newest", "middle", "oldest"]

    def test_limit_applied_after_cross_source_merge(self, patched_session_local, tmp_path):
        Session = patched_session_local
        cfg = _make_fallback_cfg(tmp_path)
        with Session() as session:
            _seed_error_log(
                session,
                error_message="db-old",
                created_at=datetime(2026, 6, 5, 12, 0, 0),
            )
            _seed_error_log(
                session,
                error_message="db-new",
                created_at=datetime(2026, 6, 5, 12, 0, 2),
            )
        _seed_fallback(cfg, error_message="fb-mid", created_at="2026-06-05T12:00:01")

        result = query_errors(self.EPOCH, limit=2, fallback_config=cfg)
        assert [r.error_message for r in result] == ["db-new", "fb-mid"]

    def test_limit_larger_than_available_returns_all(self, patched_session_local, tmp_path):
        Session = patched_session_local
        cfg = _make_fallback_cfg(tmp_path)
        with Session() as session:
            _seed_error_log(session, error_message="a")
            _seed_error_log(session, error_message="b")
        result = query_errors(self.EPOCH, limit=999, fallback_config=cfg)
        assert {r.error_message for r in result} == {"a", "b"}

    def test_filters_propagate_to_each_source(self, patched_session_local, tmp_path):
        Session = patched_session_local
        cfg = _make_fallback_cfg(tmp_path)
        with Session() as session:
            _seed_error_log(
                session,
                user_id=1,
                category="sql_execution",
                error_message="db-match",
            )
            _seed_error_log(
                session,
                user_id=2,
                category="sql_execution",
                error_message="db-other-user",
            )
            _seed_agent_run(session, user_id=1, error="agent-match")
            _seed_agent_run(session, user_id=2, error="agent-other")
        _seed_fallback(cfg, user_id=1, error_message="fb-match")
        _seed_fallback(cfg, seconds=1, user_id=2, error_message="fb-other")

        result = query_errors(
            self.EPOCH,
            user_id=1,
            fallback_config=cfg,
        )
        assert {r.error_message for r in result} == {
            "db-match",
            "agent-match",
            "fb-match",
        }

    def test_error_log_adapter_failure_isolates(self, patched_session_local, tmp_path, monkeypatch):
        Session = patched_session_local
        cfg = _make_fallback_cfg(tmp_path)
        with Session() as session:
            _seed_error_log(session, error_message="from-db")
            _seed_agent_run(session, error="from-agent")
        _seed_fallback(cfg, error_message="from-fb")

        def boom_error_log(*args, **kwargs):
            raise RuntimeError("ErrorLog adapter broken")

        monkeypatch.setattr(error_read_model, "_read_from_error_log", boom_error_log)

        result = query_errors(self.EPOCH, fallback_config=cfg)
        messages = {r.error_message for r in result}
        assert "from-agent" in messages
        assert "from-fb" in messages
        assert "from-db" not in messages

    def test_agent_run_adapter_failure_isolates(self, patched_session_local, tmp_path, monkeypatch):
        Session = patched_session_local
        cfg = _make_fallback_cfg(tmp_path)
        with Session() as session:
            _seed_error_log(session, error_message="from-db")
            _seed_agent_run(session, error="from-agent")
        _seed_fallback(cfg, error_message="from-fb")

        def boom_agent_run(*args, **kwargs):
            raise RuntimeError("AgentRun adapter broken")

        monkeypatch.setattr(error_read_model, "_read_from_agent_run", boom_agent_run)

        result = query_errors(self.EPOCH, fallback_config=cfg)
        messages = {r.error_message for r in result}
        assert "from-db" in messages
        assert "from-fb" in messages
        assert "from-agent" not in messages

    def test_fallback_adapter_failure_isolates(self, patched_session_local, tmp_path, monkeypatch):
        Session = patched_session_local
        cfg = _make_fallback_cfg(tmp_path)
        with Session() as session:
            _seed_error_log(session, error_message="from-db")
            _seed_agent_run(session, error="from-agent")
        _seed_fallback(cfg, error_message="from-fb")

        def boom_fallback(*args, **kwargs):
            raise RuntimeError("Fallback adapter broken")

        monkeypatch.setattr(error_read_model, "_read_from_fallback", boom_fallback)

        result = query_errors(self.EPOCH, fallback_config=cfg)
        messages = {r.error_message for r in result}
        assert "from-db" in messages
        assert "from-agent" in messages
        assert "from-fb" not in messages

    def test_db_session_failure_only_returns_fallback(self, patched_session_local, tmp_path, monkeypatch):
        cfg = _make_fallback_cfg(tmp_path)
        _seed_fallback(cfg, error_message="from-fb-only")

        def boom_session():
            raise RuntimeError("DB unreachable")

        monkeypatch.setattr(error_read_model, "SessionLocal", boom_session)

        result = query_errors(self.EPOCH, fallback_config=cfg)
        assert [r.error_message for r in result] == ["from-fb-only"]


# ── count_errors_by_source ────────────────────────────────────────────────


class TestCountErrorsBySource:
    EPOCH = datetime(2000, 1, 1)

    def test_returns_zero_for_all_when_empty(self, patched_session_local, tmp_path):
        cfg = _make_fallback_cfg(tmp_path)
        counts = count_errors_by_source(self.EPOCH, fallback_config=cfg)
        assert counts == {
            ErrorSource.ERROR_LOG: 0,
            ErrorSource.AGENT_RUN: 0,
            ErrorSource.FALLBACK_SINK: 0,
        }

    def test_counts_each_source_independently(self, patched_session_local, tmp_path):
        Session = patched_session_local
        cfg = _make_fallback_cfg(tmp_path)
        with Session() as session:
            for _ in range(3):
                _seed_error_log(session)
            for _ in range(2):
                _seed_agent_run(session)
            # successful run should NOT be counted
            _seed_agent_run(session, success=True, error_type=None, error=None)
        for i in range(4):
            _seed_fallback(cfg, seconds=i)

        counts = count_errors_by_source(self.EPOCH, fallback_config=cfg)
        assert counts == {
            ErrorSource.ERROR_LOG: 3,
            ErrorSource.AGENT_RUN: 2,
            ErrorSource.FALLBACK_SINK: 4,
        }

    def test_time_range_applied(self, patched_session_local, tmp_path):
        Session = patched_session_local
        cfg = _make_fallback_cfg(tmp_path)
        with Session() as session:
            _seed_error_log(
                session,
                error_message="old",
                created_at=datetime(2025, 1, 1),
            )
            _seed_error_log(
                session,
                error_message="new",
                created_at=datetime(2026, 6, 5),
            )
        counts = count_errors_by_source(datetime(2026, 1, 1), fallback_config=cfg)
        assert counts[ErrorSource.ERROR_LOG] == 1

    def test_db_session_failure_returns_zero_for_db_sources(self, patched_session_local, tmp_path, monkeypatch):
        cfg = _make_fallback_cfg(tmp_path)
        _seed_fallback(cfg)

        def boom_session():
            raise RuntimeError("DB unreachable")

        monkeypatch.setattr(error_read_model, "SessionLocal", boom_session)

        counts = count_errors_by_source(self.EPOCH, fallback_config=cfg)
        assert counts[ErrorSource.ERROR_LOG] == 0
        assert counts[ErrorSource.AGENT_RUN] == 0
        # Fallback still works since it doesn't use SessionLocal
        assert counts[ErrorSource.FALLBACK_SINK] == 1
