"""Tests for views.errors_helpers (Phase 1: pure helpers)."""

from __future__ import annotations

import datetime as dt

import pandas as pd

from orm.error_read_model import DEFAULT_SOURCES, ErrorRow, ErrorSource
from views.errors_helpers import (
    _csv_to_set,
    _export_filename,
    _format_results_table,
    _parse_user_id,
    _pretty_context_data,
    _query_filtered,
    _resolve_sources,
    _time_range_to_since,
)


def _make_row(**overrides) -> ErrorRow:
    base = dict(
        id="error_log:1",
        source=ErrorSource.ERROR_LOG,
        created_at=dt.datetime(2026, 6, 5, 12, 0, 0),
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


# ── _time_range_to_since ──────────────────────────────────────────────────


class TestTimeRangeToSince:
    def _within(self, when: dt.datetime, expected: dt.datetime, tolerance_s: int = 5) -> bool:
        return abs((when - expected).total_seconds()) <= tolerance_s

    def test_7_days(self):
        result = _time_range_to_since(7)
        expected = dt.datetime.now() - dt.timedelta(days=7)
        assert self._within(result, expected)

    def test_30_days(self):
        result = _time_range_to_since(30)
        expected = dt.datetime.now() - dt.timedelta(days=30)
        assert self._within(result, expected)

    def test_90_days(self):
        result = _time_range_to_since(90)
        expected = dt.datetime.now() - dt.timedelta(days=90)
        assert self._within(result, expected)


# ── _resolve_sources ──────────────────────────────────────────────────────


class TestResolveSources:
    def test_empty_csv_returns_default_sources(self):
        assert _resolve_sources("") == set(DEFAULT_SOURCES)

    def test_one_value_returns_subset(self):
        assert _resolve_sources("error_log") == {ErrorSource.ERROR_LOG}

    def test_two_values(self):
        assert _resolve_sources("error_log,agent_run") == {
            ErrorSource.ERROR_LOG,
            ErrorSource.AGENT_RUN,
        }

    def test_unknown_value_is_silently_ignored(self):
        # Defensive: an unknown source from a corrupted cache key should not raise.
        assert _resolve_sources("error_log,bogus") == {ErrorSource.ERROR_LOG}

    def test_whitespace_tolerated(self):
        assert _resolve_sources(" error_log , agent_run ") == {
            ErrorSource.ERROR_LOG,
            ErrorSource.AGENT_RUN,
        }


# ── _format_results_table ─────────────────────────────────────────────────


class TestFormatResultsTable:
    EXPECTED_COLUMNS = [
        "created_at",
        "source",
        "category",
        "severity",
        "error_type",
        "error_message",
    ]

    def test_empty_input_returns_empty_dataframe_with_columns(self):
        df = _format_results_table([])
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == self.EXPECTED_COLUMNS
        assert len(df) == 0

    def test_populated_full_field_round_trip(self):
        rows = [
            _make_row(
                source=ErrorSource.AGENT_RUN,
                category="agent_run",
                severity="warning",
                error_type="ToolCapReached",
                error_message="too many tool calls",
            ),
            _make_row(
                source=ErrorSource.FALLBACK_SINK,
                category="sql_execution",
                severity="error",
                error_type="RuntimeError",
                error_message="boom",
            ),
        ]
        df = _format_results_table(rows)
        assert list(df.columns) == self.EXPECTED_COLUMNS
        assert len(df) == 2
        # Enum → string value
        assert list(df["source"]) == ["agent_run", "fallback_sink"]
        assert list(df["category"]) == ["agent_run", "sql_execution"]
        assert list(df["severity"]) == ["warning", "error"]
        assert list(df["error_type"]) == ["ToolCapReached", "RuntimeError"]
        assert list(df["error_message"]) == ["too many tool calls", "boom"]

    def test_none_optionals_become_empty_strings(self):
        rows = [_make_row()]
        df = _format_results_table(rows)
        assert df["category"][0] == ""
        assert df["severity"][0] == ""
        assert df["error_type"][0] == ""
        assert df["error_message"][0] == ""
        # source still serializes (required field, never None)
        assert df["source"][0] == "error_log"


# ── _csv_to_set ───────────────────────────────────────────────────────────


class TestCsvToSet:
    def test_empty_returns_none(self):
        assert _csv_to_set("") is None

    def test_whitespace_only_returns_none(self):
        assert _csv_to_set("   ") is None

    def test_single_value(self):
        assert _csv_to_set("sql_execution") == {"sql_execution"}

    def test_multiple_values(self):
        assert _csv_to_set("a,b,c") == {"a", "b", "c"}

    def test_strips_whitespace(self):
        assert _csv_to_set(" a , b ") == {"a", "b"}

    def test_empty_tokens_filtered_out(self):
        assert _csv_to_set("a,,b,") == {"a", "b"}


# ── _parse_user_id ────────────────────────────────────────────────────────


class TestParseUserId:
    def test_empty_returns_none(self):
        assert _parse_user_id("") is None

    def test_whitespace_only_returns_none(self):
        assert _parse_user_id("   ") is None

    def test_valid_int_returns_int(self):
        assert _parse_user_id("42") == 42

    def test_whitespace_padded_int(self):
        assert _parse_user_id("  7  ") == 7

    def test_zero_is_allowed(self):
        assert _parse_user_id("0") == 0

    def test_negative_returns_none(self):
        assert _parse_user_id("-5") is None

    def test_non_numeric_returns_none(self):
        assert _parse_user_id("abc") is None

    def test_decimal_returns_none(self):
        assert _parse_user_id("3.14") is None


# ── _pretty_context_data ──────────────────────────────────────────────────


class TestPrettyContextData:
    def test_none_returns_none(self):
        assert _pretty_context_data(None) is None

    def test_empty_returns_none(self):
        assert _pretty_context_data("") is None

    def test_whitespace_only_returns_none(self):
        assert _pretty_context_data("   ") is None

    def test_valid_json_object_pretty_printed(self):
        result = _pretty_context_data('{"a": 1, "b": "x"}')
        assert result is not None
        import json as _json

        # Round-trips back to the same dict
        assert _json.loads(result) == {"a": 1, "b": "x"}
        # And is indented (more than one line)
        assert "\n" in result

    def test_valid_json_array_pretty_printed(self):
        result = _pretty_context_data("[1, 2, 3]")
        assert result is not None
        assert "\n" in result

    def test_invalid_json_returns_raw_text(self):
        raw = "not valid json {{"
        assert _pretty_context_data(raw) == raw


# ── _export_filename ──────────────────────────────────────────────────────


class TestExportFilename:
    def test_ends_with_json_extension(self):
        name = _export_filename(dt.datetime(2026, 6, 5), "error_log")
        assert name.endswith(".json")

    def test_includes_since_iso(self):
        name = _export_filename(dt.datetime(2026, 6, 5, 12, 0, 0), "error_log")
        assert "20260605T120000" in name

    def test_explicit_until_overrides_now(self):
        name = _export_filename(
            dt.datetime(2026, 6, 5),
            "error_log",
            until=dt.datetime(2026, 7, 1, 9, 30, 45),
        )
        assert "20260701T093045" in name

    def test_empty_sources_csv_becomes_all(self):
        name = _export_filename(dt.datetime(2026, 6, 5), "")
        assert "all" in name

    def test_multiple_sources_joined_with_dash(self):
        name = _export_filename(dt.datetime(2026, 6, 5), "error_log,agent_run")
        assert "error_log-agent_run" in name

    def test_single_source_preserved(self):
        name = _export_filename(dt.datetime(2026, 6, 5), "fallback_sink")
        assert "fallback_sink" in name


# ── _query_filtered smoke test (cross-source) ─────────────────────────────


class TestQueryFilteredSmoke:
    """End-to-end shape check that the view's data helper composes the read
    model + fallback adapter correctly. Uses an in-memory SQLite plus a
    tmp JSONL fallback file. Mirrors the pattern in
    tests/orm/test_error_read_model.py without re-testing the read-model
    semantics — just verifies the view's composition."""

    def test_view_helper_returns_rows_from_all_three_sources(self, tmp_path, monkeypatch):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        from orm import error_read_model
        from orm.models import AgentRun, Base, ErrorLog
        from utils.error_fallback_sink import (
            ErrorLoggingConfig,
            _reset_handler_cache,
            write_fallback_record,
        )

        _reset_handler_cache()

        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine, autocommit=False, autoflush=False)
        monkeypatch.setattr(error_read_model, "SessionLocal", Session)

        with Session() as session:
            session.add(
                ErrorLog(
                    category="sql_execution",
                    severity="error",
                    error_type="RuntimeError",
                    error_message="from-error-log",
                    created_at=dt.datetime.now() - dt.timedelta(hours=1),
                )
            )
            session.add(
                AgentRun(
                    run_id="smoke-run-1",
                    session_id="smoke-sess",
                    user_id=1,
                    user_role=0,
                    status="error",
                    success=False,
                    error_type="ToolError",
                    error="from-agent-run",
                    created_at=dt.datetime.now() - dt.timedelta(hours=2),
                )
            )
            session.commit()

        cfg = ErrorLoggingConfig(
            fallback_path=tmp_path / "smoke.jsonl",
            fallback_max_bytes=5_000_000,
            fallback_backup_count=5,
        )
        write_fallback_record(
            {
                "created_at": (dt.datetime.now() - dt.timedelta(hours=3)).isoformat(),
                "category": "sql_execution",
                "severity": "error",
                "error_type": "FallbackError",
                "error_message": "from-fallback",
                "user_id": 1,
            },
            config=cfg,
        )

        rows_dicts, counts = _query_filtered(
            days=7,
            sources_csv="",  # default = all three
            categories_csv="",
            severities_csv="",
            search="",
            user_id_text="",
            fallback_config=cfg,
        )

        # All three sources present
        messages = {r["error_message"] for r in rows_dicts}
        assert "from-error-log" in messages
        assert "from-agent-run" in messages
        assert "from-fallback" in messages

        # Counts include each source
        assert counts["error_log"] >= 1
        assert counts["agent_run"] >= 1
        assert counts["fallback_sink"] >= 1

        # Sorted descending by created_at (ErrorLog is newest)
        assert rows_dicts[0]["error_message"] == "from-error-log"

        # Every payload is JSON-serializable (no Enum, no datetime objects)
        import json as _json

        _json.dumps(rows_dicts)

    def test_view_helper_respects_filters(self, tmp_path, monkeypatch):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        from orm import error_read_model
        from orm.models import Base, ErrorLog
        from utils.error_fallback_sink import (
            ErrorLoggingConfig,
            _reset_handler_cache,
        )

        _reset_handler_cache()

        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine, autocommit=False, autoflush=False)
        monkeypatch.setattr(error_read_model, "SessionLocal", Session)

        with Session() as session:
            for cat, msg in [
                ("sql_execution", "match"),
                ("chart_generation", "skip-1"),
                ("sql_execution", "skip-2"),
            ]:
                session.add(
                    ErrorLog(
                        category=cat,
                        severity="error",
                        error_type="RuntimeError",
                        error_message=msg,
                        created_at=dt.datetime.now() - dt.timedelta(hours=1),
                    )
                )
            session.commit()

        cfg = ErrorLoggingConfig(
            fallback_path=tmp_path / "x.jsonl",
            fallback_max_bytes=5_000_000,
            fallback_backup_count=5,
        )

        rows_dicts, _ = _query_filtered(
            days=7,
            sources_csv="error_log",
            categories_csv="sql_execution",
            severities_csv="",
            search="match",
            user_id_text="",
            fallback_config=cfg,
        )
        assert [r["error_message"] for r in rows_dicts] == ["match"]
