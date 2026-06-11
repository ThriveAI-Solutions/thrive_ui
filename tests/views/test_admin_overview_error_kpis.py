"""Tests for the Admin Overview error-count KPIs after the two-ledger
disambiguation (Epic #161, Feature Spec #162).

Asserts:

- The KPI labels are renamed to "Chat Errors" and "Critical System
  Errors" (the old "Errors" / "Critical Errors" labels are gone).
- "Critical System Errors" is fed by ``views.errors._load_aggregates``
  (the 3-source union via :func:`query_aggregates`) — NOT by the legacy
  ``orm.logging_functions.get_error_stats`` ErrorLog-only counter.
- The shared cache is honoured: rendering both the Errors tab and the
  Admin Overview within one session calls ``query_aggregates`` exactly
  once per ``days_int`` (because ``_load_aggregates`` is
  ``@st.cache_data``-decorated and shared across modules).
- Module-level two-ledger contract docstrings exist on both
  ``views.admin_analytics`` and ``views.errors``.
"""

from __future__ import annotations

import datetime as dt
import importlib
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Streamlit recorder — mirrors the pattern in test_admin_actions_dialog.py
# ---------------------------------------------------------------------------


class _Column:
    """Streamlit ``st.columns(...)`` element with a ``with`` context."""

    def __init__(self, recorder, idx):
        self._recorder = recorder
        self._idx = idx

    def __enter__(self):
        self._recorder.current_column = self._idx
        return self

    def __exit__(self, exc_type, exc_val, tb):
        self._recorder.current_column = None
        return False


class _Container:
    def __init__(self, recorder):
        self._recorder = recorder

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, tb):
        return False


class _RecordingStreamlit:
    """A minimal stand-in for the ``streamlit`` module for KPI-row rendering.

    Records every ``_kpi_card(...)`` invocation by intercepting the
    streamlit primitives it calls — ``container`` (for the bordered
    box), ``markdown`` (for the label + value), and ``caption`` (for
    the help text).
    """

    def __init__(self):
        self.session_state: dict = {}
        self.current_column: int | None = None
        # Per-column markdown / caption text, in the order each was called.
        self.markdown_calls: list[tuple[int | None, str]] = []
        self.caption_calls: list[tuple[int | None, str]] = []
        self.divider_count = 0

    def container(self, *_a, **_kw):
        return _Container(self)

    def columns(self, n, **_kw):
        if isinstance(n, list):
            return [_Column(self, i) for i in range(len(n))]
        return [_Column(self, i) for i in range(int(n))]

    def markdown(self, body, **_kwargs):
        self.markdown_calls.append((self.current_column, str(body)))

    def caption(self, body, **_kwargs):
        self.caption_calls.append((self.current_column, str(body)))

    def divider(self):
        self.divider_count += 1

    # Surface stubs — the test never asserts on these but the
    # production code calls into them.
    def plotly_chart(self, *_a, **_kw):
        pass

    def dataframe(self, *_a, **_kw):
        pass

    def info(self, *_a, **_kw):
        pass

    def subheader(self, *_a, **_kw):
        pass

    def write(self, *_a, **_kw):
        pass

    def expander(self, *_a, **_kw):
        return _Container(self)

    def button(self, *_a, **_kw):
        return False

    def success(self, *_a, **_kw):
        pass

    def error(self, *_a, **_kw):
        pass

    def text_input(self, _label, value="", **_kw):
        return value

    def multiselect(self, _label, options, default=None, **_kw):
        return list(default) if default else list(options)

    def download_button(self, *_a, **_kw):
        return False

    def code(self, *_a, **_kw):
        pass

    def page_link(self, *_a, **_kw):
        pass

    def rerun(self):
        pass

    def stop(self):
        pass


# ---------------------------------------------------------------------------
# _read_metrics stub — production code unpacks a 6-tuple
# ---------------------------------------------------------------------------


def _stub_read_metrics(days=30):  # noqa: ARG001 — production code calls with kw
    _ = days
    Row = SimpleNamespace
    rows = [
        Row(
            d=dt.date.today().strftime("%Y-%m-%d"),
            questions=2,
            charts=1,
            summaries=1,
            sql=2,
            errors=3,
        ),
    ]
    result_rows = [Row(d=rows[0].d, questions=2, results=2)]
    overall_stats = {"avg": 0.0, "min": 0.0, "max": 0.0, "stddev": 0.0, "median": 0.0, "n": 0}
    perf_types = {
        "sql": overall_stats,
        "summary": overall_stats,
        "chart": overall_stats,
        "dataframe": overall_stats,
    }
    return (5, 2, rows, result_rows, overall_stats, perf_types)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOverviewErrorKpiLabels:
    """The label rename — old labels are gone, new labels are present."""

    def _render(self, monkeypatch, *, aggregates_critical=4):
        from views import admin_analytics, errors

        rec = _RecordingStreamlit()
        monkeypatch.setattr(admin_analytics, "st", rec)
        monkeypatch.setattr(errors, "st", rec)
        monkeypatch.setattr(admin_analytics, "_read_metrics", _stub_read_metrics)
        # Strip the @st.cache_data wrapper from _load_aggregates so the
        # function-level patch on query_aggregates flows through.
        agg_dict = {
            "total": 47,
            "critical": aggregates_critical,
            "sql_errors": 8,
            "retry_success_rate": 75.0,
            "over_time_by_category": [],
            "by_category": {},
            "by_severity": [],
        }

        def _fake_load_aggregates(days):
            return agg_dict

        monkeypatch.setattr(errors, "_load_aggregates", _fake_load_aggregates)

        with patch("orm.logging_functions.get_llm_stats", return_value={"total": 10, "avg_latency_ms": 250.0}):
            with patch("orm.logging_functions.get_activity_stats", return_value={"logins_today": 3}):
                # _render_overview_tab does substantial DB work AFTER the
                # KPI row (latency distribution + question audit preview).
                # We only care about the KPI markdown calls; suppress
                # downstream DB failures so the assertion can run.
                try:
                    admin_analytics._render_overview_tab(30)
                except Exception:
                    pass
        return rec

    def test_chat_errors_label_present(self, monkeypatch):
        rec = self._render(monkeypatch)
        labels = [text for (_, text) in rec.markdown_calls]
        # Bold KPI label markup is `**<label>**`
        assert any("**Chat Errors**" == t for t in labels), (
            f"Expected exactly '**Chat Errors**' KPI label among markdown calls: {labels}"
        )

    def test_old_errors_label_absent(self, monkeypatch):
        rec = self._render(monkeypatch)
        labels = [text for (_, text) in rec.markdown_calls]
        # The exact-match old label is gone (matches "**Errors**" but not "**Chat Errors**").
        assert not any(t == "**Errors**" for t in labels), (
            f"Old label '**Errors**' should be retired but appears in markdown calls: {labels}"
        )

    def test_critical_system_errors_label_present(self, monkeypatch):
        rec = self._render(monkeypatch)
        labels = [text for (_, text) in rec.markdown_calls]
        assert any("**Critical System Errors**" == t for t in labels), (
            f"Expected exactly '**Critical System Errors**' label: {labels}"
        )

    def test_old_critical_errors_label_absent(self, monkeypatch):
        rec = self._render(monkeypatch)
        labels = [text for (_, text) in rec.markdown_calls]
        assert not any(t == "**Critical Errors**" for t in labels), (
            f"Old label '**Critical Errors**' should be retired: {labels}"
        )


class TestCriticalSystemErrorsValue:
    """Counter rewire — value comes from _load_aggregates['critical']."""

    def test_critical_system_errors_value_matches_aggregates(self, monkeypatch):
        from views import admin_analytics, errors

        rec = _RecordingStreamlit()
        monkeypatch.setattr(admin_analytics, "st", rec)
        monkeypatch.setattr(errors, "st", rec)
        monkeypatch.setattr(admin_analytics, "_read_metrics", _stub_read_metrics)

        # The seeded aggregates result — critical=12 should bubble through.
        monkeypatch.setattr(
            errors,
            "_load_aggregates",
            lambda days: {
                "total": 47,
                "critical": 12,
                "sql_errors": 8,
                "retry_success_rate": 75.0,
                "over_time_by_category": [],
                "by_category": {},
                "by_severity": [],
            },
        )

        with patch("orm.logging_functions.get_llm_stats", return_value={"total": 10, "avg_latency_ms": 250.0}):
            with patch("orm.logging_functions.get_activity_stats", return_value={"logins_today": 3}):
                try:
                    admin_analytics._render_overview_tab(30)
                except Exception:
                    pass

        # Find the markdown call for the value rendered immediately after
        # the "**Critical System Errors**" label.
        idxs = [i for i, (_, t) in enumerate(rec.markdown_calls) if t == "**Critical System Errors**"]
        assert idxs, "Critical System Errors label was not rendered"
        value_text = rec.markdown_calls[idxs[0] + 1][1]
        # _kpi_card renders the value as: <h3 style='margin-top:0'>12</h3>
        assert ">12<" in value_text, f"Expected critical=12 in value markup, got: {value_text}"


class TestCriticalSystemErrorsIndependentOfGetErrorStats:
    """If something still imported get_error_stats and returned a wildly
    different number, the KPI must NOT pick it up. The rewire severs the
    coupling — get_error_stats is gone, but if someone re-adds it, this
    test catches the leak.
    """

    def test_kpi_ignores_get_error_stats_returns(self, monkeypatch):
        from views import admin_analytics, errors

        rec = _RecordingStreamlit()
        monkeypatch.setattr(admin_analytics, "st", rec)
        monkeypatch.setattr(errors, "st", rec)
        monkeypatch.setattr(admin_analytics, "_read_metrics", _stub_read_metrics)
        monkeypatch.setattr(
            errors,
            "_load_aggregates",
            lambda days: {
                "total": 1,
                "critical": 7,
                "sql_errors": 0,
                "retry_success_rate": 0.0,
                "over_time_by_category": [],
                "by_category": {},
                "by_severity": [],
            },
        )

        # If a stray get_error_stats exists, mock it to a clearly distinct value.
        with patch("orm.logging_functions.get_llm_stats", return_value={"total": 0, "avg_latency_ms": 0.0}):
            with patch("orm.logging_functions.get_activity_stats", return_value={"logins_today": 0}):
                # get_error_stats is no longer imported by admin_analytics, but
                # we add the patch to confirm the KPI doesn't read from it.
                if hasattr(
                    importlib.import_module("orm.logging_functions"),
                    "get_error_stats",
                ):
                    with patch(
                        "orm.logging_functions.get_error_stats",
                        return_value={
                            "total": 99999,
                            "critical": 99999,
                            "sql_errors": 99999,
                            "retry_success_rate": 0.0,
                        },
                    ):
                        try:
                            admin_analytics._render_overview_tab(30)
                        except Exception:
                            pass
                else:
                    try:
                        admin_analytics._render_overview_tab(30)
                    except Exception:
                        pass

        idxs = [i for i, (_, t) in enumerate(rec.markdown_calls) if t == "**Critical System Errors**"]
        assert idxs, "Critical System Errors label was not rendered"
        value_text = rec.markdown_calls[idxs[0] + 1][1]
        assert ">7<" in value_text, (
            f"KPI should reflect _load_aggregates['critical']=7, not get_error_stats; got: {value_text}"
        )


class TestGetErrorStatsRetired:
    """The legacy ErrorLog-only counter no longer powers any production
    code path. Per the spec, since no tests pin it and no production code
    references it, it is deleted from orm/logging_functions.py."""

    def test_get_error_stats_no_longer_exported(self):
        import orm.logging_functions as lf

        # The function is deleted entirely.
        assert not hasattr(lf, "get_error_stats"), (
            "get_error_stats should be deleted from orm.logging_functions "
            "after Feature Spec #162 retires the legacy ErrorLog-only counter."
        )

    def test_get_error_stats_not_referenced_by_admin_analytics(self):
        import inspect

        from views import admin_analytics

        src = inspect.getsource(admin_analytics)
        assert "get_error_stats" not in src, (
            "views.admin_analytics must not reference get_error_stats — "
            "the Critical System Errors KPI routes via _load_aggregates instead."
        )


class TestSharedCacheAcrossOverviewAndErrorsTab:
    """Cross-surface invariant: rendering both surfaces in one session
    only calls ``query_aggregates`` once per ``days_int``.

    This is the load-bearing contract that the Overview KPI and the
    Errors-tab Critical card always agree under the same time range.
    """

    def test_overview_imports_shared_load_aggregates(self):
        """Overview must reach Ledger B via the shared cache (the same
        ``_load_aggregates`` the Errors tab uses) — not via a sibling
        loader. The structural defense for cross-surface agreement.
        """
        import inspect

        from views import admin_analytics

        src = inspect.getsource(admin_analytics._render_overview_tab)
        assert "from views.errors import _load_aggregates" in src, (
            "Overview must import _load_aggregates from views.errors so the "
            "Critical System Errors KPI shares the cache with the Errors tab. "
            "See views/admin_analytics.py:1347 for precedent."
        )

    def test_query_aggregates_called_once_per_days_int(self, monkeypatch):
        """Two consecutive calls to ``_load_aggregates(30)`` (one from
        each surface in production) result in exactly one call to the
        underlying ``query_aggregates`` — the second call hits the
        ``@st.cache_data`` store. This is the shared-cache contract.
        """
        from views import errors

        # Spy on the underlying query_aggregates call.
        from orm.error_read_model import ErrorAggregates

        fake_agg = ErrorAggregates(
            total=10,
            critical=2,
            sql_errors=1,
            retry_attempted=0,
            retry_successful=0,
            retry_success_rate=0.0,
            over_time_by_category=[],
            by_category={},
            by_severity=[],
        )
        call_counter = MagicMock(return_value=fake_agg)
        monkeypatch.setattr(errors, "query_aggregates", call_counter)

        # Clear any pre-existing cache so the baseline is deterministic.
        errors._load_aggregates.clear()

        # Simulate the two surfaces (Overview + Errors tab) loading
        # aggregates under the same days_int. The Streamlit cache is
        # process-wide and persists across module imports.
        first = errors._load_aggregates(30)
        second = errors._load_aggregates(30)

        assert first == second
        assert call_counter.call_count == 1, (
            f"Expected query_aggregates to be called exactly once for "
            f"days_int=30 across both surfaces (shared cache); got "
            f"{call_counter.call_count} calls. Shared-cache contract is broken."
        )

        errors._load_aggregates.clear()


class TestModuleDocstringTwoLedgerContract:
    """Both module docstrings must declare the two-ledger contract."""

    @pytest.mark.parametrize(
        "module_name",
        ["views.admin_analytics", "views.errors"],
    )
    def test_module_docstring_documents_two_ledger_contract(self, module_name):
        mod = importlib.import_module(module_name)
        doc = mod.__doc__ or ""
        assert "Ledger A" in doc, f"{module_name} must document Ledger A (chat-flow errors)."
        assert "Ledger B" in doc, f"{module_name} must document Ledger B (system errors)."
