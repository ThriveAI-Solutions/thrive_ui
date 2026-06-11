"""Tests for the Total-vs-subset visual treatment in the Errors-tab
Analytics KPI row (Epic #161, Feature Spec #164).

The four-card row renders as:

- Total Errors    — primary (existing _kpi_card; additive style).
- Critical        — subset (new _render_subset_kpi_card; "of N (Z.Z%)").
- SQL Errors      — subset (new _render_subset_kpi_card; "of N (Z.Z%)").
- Retry Success%  — rate (existing _kpi_card with % suffix).

The subset cards textually signal that their counts are *slices* of
Total, not addends. The Analytics caption adds explicit prose that the
two subsets may overlap each other (a critical SQL-generation error
counts toward both).

All assertions exercise the pure-format helper
:func:`views.errors._subset_kpi_caption_text` so the tests run without
a Streamlit script context.
"""

from __future__ import annotations

from unittest.mock import patch

from views.errors import _subset_kpi_caption_text


# ---------------------------------------------------------------------------
# Pure-format helper — caption text
# ---------------------------------------------------------------------------


class TestSubsetCaptionShape:
    """``_subset_kpi_caption_text(count, total)`` returns the denominator
    text only — the count itself is rendered as the primary value by
    the wrapper.
    """

    def test_basic_subset_denominator(self):
        assert _subset_kpi_caption_text(5, 47) == "of 47 (10.6%)"

    def test_zero_count_renders_zero_percent(self):
        assert _subset_kpi_caption_text(0, 47) == "of 47 (0.0%)"

    def test_count_equals_total_renders_100_percent(self):
        assert _subset_kpi_caption_text(47, 47) == "of 47 (100.0%)"

    def test_zero_total_omits_percentage_no_divide_by_zero(self):
        # Empty time range — render "of 0" without a % suffix and
        # without raising.
        assert _subset_kpi_caption_text(0, 0) == "of 0"

    def test_one_of_three_renders_33_3_percent(self):
        # round(100/3, 1) == 33.3
        assert _subset_kpi_caption_text(1, 3) == "of 3 (33.3%)"

    def test_two_of_three_renders_66_7_percent(self):
        # round(200/3, 1) == 66.7
        assert _subset_kpi_caption_text(2, 3) == "of 3 (66.7%)"

    def test_large_total_one_decimal_precision(self):
        # 123 / 10000 = 0.0123 → 1.23% → rounded to 1.2%
        assert _subset_kpi_caption_text(123, 10000) == "of 10000 (1.2%)"


# ---------------------------------------------------------------------------
# Wrapper — _render_subset_kpi_card delegates correctly
# ---------------------------------------------------------------------------


class _Container:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class _RecorderSt:
    def __init__(self):
        self.markdown_calls: list[str] = []
        self.caption_calls: list[str] = []

    def container(self, *_a, **_kw):
        return _Container()

    def markdown(self, body, **_kw):
        self.markdown_calls.append(str(body))

    def caption(self, body, **_kw):
        self.caption_calls.append(str(body))


class TestRenderSubsetKpiCardWrapper:
    def test_renders_label_value_and_denominator(self, monkeypatch):
        from views import errors

        rec = _RecorderSt()
        monkeypatch.setattr(errors, "st", rec)

        errors._render_subset_kpi_card("Critical", 5, 47, help_text="Severity = critical")

        # Label and value rendered via markdown.
        assert "**Critical**" in rec.markdown_calls
        assert any(">5<" in m for m in rec.markdown_calls), (
            f"Expected the count 5 to appear as a <h3> value; got: {rec.markdown_calls}"
        )

        # Caption carries the denominator + help_text.
        assert "of 47 (10.6%)" in rec.caption_calls
        assert "Severity = critical" in rec.caption_calls

    def test_zero_total_caption_omits_percentage(self, monkeypatch):
        from views import errors

        rec = _RecorderSt()
        monkeypatch.setattr(errors, "st", rec)

        errors._render_subset_kpi_card("Critical", 0, 0)
        assert "of 0" in rec.caption_calls
        # No percentage rendered for total=0.
        assert not any("%" in c for c in rec.caption_calls), (
            f"Caption must not include a percentage when total=0; got: {rec.caption_calls}"
        )

    def test_help_text_optional(self, monkeypatch):
        from views import errors

        rec = _RecorderSt()
        monkeypatch.setattr(errors, "st", rec)

        errors._render_subset_kpi_card("SQL Errors", 8, 47)
        # Denominator caption is present; no extra caption beyond it.
        assert rec.caption_calls == ["of 47 (17.0%)"]


# ---------------------------------------------------------------------------
# Integration — helper-invocation spying in views.errors.render(...)
# ---------------------------------------------------------------------------


class TestAnalyticsRowDelegatesToCorrectHelpers:
    """Inside the Analytics row, Total Errors and Retry Success continue
    to render via the existing ``_kpi_card`` (additive / rate style);
    Critical and SQL Errors render via the new ``_render_subset_kpi_card``
    (subset style). Verified by patching both helpers and asserting their
    call signatures.
    """

    def _render_with_spies(self, monkeypatch, *, total=47, critical=5, sql_errors=8):
        """Run views.errors.render with helpers patched; return the call lists."""
        from views import errors

        # Stub Streamlit primitives the render path touches so it
        # completes without a Streamlit runtime.
        from tests.views.test_admin_overview_error_kpis import _RecordingStreamlit

        rec = _RecordingStreamlit()
        monkeypatch.setattr(errors, "st", rec)

        # Bypass the @st.cache_data layer by patching _load_aggregates
        # directly. _load is also cached — patch it to return empty.
        monkeypatch.setattr(
            errors,
            "_load_aggregates",
            lambda days: {
                "total": total,
                "critical": critical,
                "sql_errors": sql_errors,
                "retry_success_rate": 75.0,
                "over_time_by_category": [],
                "by_category": {},
                "by_severity": [],
            },
        )
        monkeypatch.setattr(errors, "_load", lambda *_a, **_kw: ([], {}))

        with patch.object(errors, "_kpi_card") as kpi_mock:
            with patch.object(errors, "_render_subset_kpi_card") as subset_mock:
                # Disable the equation render so we don't get extra calls
                # tangled with the assertion — it's tested in #163.
                with patch.object(errors, "_render_sources_equation"):
                    try:
                        errors.render(30)
                    except Exception:
                        # render() does substantial downstream work after
                        # the KPI row (charts, expanders, etc); we only
                        # need the helper-invocation calls to be recorded.
                        pass
        return kpi_mock, subset_mock

    def test_total_errors_uses_kpi_card(self, monkeypatch):
        kpi_mock, _ = self._render_with_spies(monkeypatch)
        labels = [call.args[0] for call in kpi_mock.call_args_list if call.args]
        assert "Total Errors" in labels, (
            f"Total Errors must continue to render via _kpi_card (additive style); _kpi_card was called for: {labels}"
        )

    def test_retry_success_uses_kpi_card(self, monkeypatch):
        kpi_mock, _ = self._render_with_spies(monkeypatch)
        labels = [call.args[0] for call in kpi_mock.call_args_list if call.args]
        assert "Retry Success" in labels, (
            f"Retry Success must continue to render via _kpi_card (rate style); got: {labels}"
        )

    def test_critical_uses_subset_kpi_card(self, monkeypatch):
        _, subset_mock = self._render_with_spies(monkeypatch)
        labels = [call.args[0] for call in subset_mock.call_args_list if call.args]
        assert "Critical" in labels, f"Critical must render via _render_subset_kpi_card (subset style); got: {labels}"

    def test_sql_errors_uses_subset_kpi_card(self, monkeypatch):
        _, subset_mock = self._render_with_spies(monkeypatch)
        labels = [call.args[0] for call in subset_mock.call_args_list if call.args]
        assert "SQL Errors" in labels, (
            f"SQL Errors must render via _render_subset_kpi_card (subset style); got: {labels}"
        )

    def test_critical_not_rendered_via_kpi_card(self, monkeypatch):
        """The Critical card must NOT also be sent through _kpi_card —
        the subset visual treatment is the structural differentiator."""
        kpi_mock, _ = self._render_with_spies(monkeypatch)
        labels = [call.args[0] for call in kpi_mock.call_args_list if call.args]
        assert "Critical" not in labels, f"Critical must NOT render via _kpi_card after #164; got: {labels}"

    def test_sql_errors_not_rendered_via_kpi_card(self, monkeypatch):
        kpi_mock, _ = self._render_with_spies(monkeypatch)
        labels = [call.args[0] for call in kpi_mock.call_args_list if call.args]
        assert "SQL Errors" not in labels, f"SQL Errors must NOT render via _kpi_card after #164; got: {labels}"

    def test_subset_helper_receives_count_and_total(self, monkeypatch):
        """The subset helper gets (count, total) so the denominator
        renders correctly."""
        _, subset_mock = self._render_with_spies(monkeypatch, total=47, critical=5, sql_errors=8)
        # Build a map of label → (count, total) from the recorded calls.
        rendered = {}
        for call in subset_mock.call_args_list:
            args = call.args
            # Signature: _render_subset_kpi_card(label, count, total, help_text=...)
            if len(args) >= 3:
                rendered[args[0]] = (args[1], args[2])
        assert rendered.get("Critical") == (5, 47), (
            f"Critical must get (count=5, total=47); got: {rendered.get('Critical')}"
        )
        assert rendered.get("SQL Errors") == (8, 47), (
            f"SQL Errors must get (count=8, total=47); got: {rendered.get('SQL Errors')}"
        )


# ---------------------------------------------------------------------------
# Caption disambiguation — admins are told the subsets may overlap
# ---------------------------------------------------------------------------


class TestAnalyticsCaptionDisambiguation:
    """The Analytics block caption explicitly states that Critical and
    SQL Errors are subsets that may overlap each other. Without this,
    admins eventually try to partition Total and get confused.
    """

    def test_caption_explains_subset_relationship(self, monkeypatch):
        from views import errors

        from tests.views.test_admin_overview_error_kpis import _RecordingStreamlit

        rec = _RecordingStreamlit()
        monkeypatch.setattr(errors, "st", rec)
        monkeypatch.setattr(
            errors,
            "_load_aggregates",
            lambda days: {
                "total": 0,
                "critical": 0,
                "sql_errors": 0,
                "retry_success_rate": 0.0,
                "over_time_by_category": [],
                "by_category": {},
                "by_severity": [],
            },
        )
        monkeypatch.setattr(errors, "_load", lambda *_a, **_kw: ([], {}))
        try:
            errors.render(30)
        except Exception:
            pass

        captions = [body for (_, body) in rec.caption_calls]
        # The Analytics caption must declare the subsets-of-Total relationship.
        assert any("subsets of Total Errors" in c for c in captions), (
            f"Analytics caption must declare that Critical and SQL Errors are "
            f"subsets of Total Errors. Captions found: {captions}"
        )
        # And that the two subsets may overlap each other.
        assert any("overlap" in c.lower() for c in captions), (
            f"Analytics caption must explain the overlap (a critical "
            f"SQL-generation error counts toward both). Captions: {captions}"
        )
