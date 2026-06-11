"""Tests for the Sources-row additivity equation on the Errors tab
(Epic #161, Feature Spec #163).

The equation renders immediately below the three Source KPI cards as::

    Total: X = Error Log (A) + Agent Runs (B) + Fallback File (C)

When a Source chip is deselected, that source's term wraps in markdown
strike-through (``~~term~~``) but Total is unchanged from the unfiltered
cross-source sum. This preserves the invariant
*Sources equation Total == Analytics row Total Errors KPI* — the
load-bearing Epic-level guarantee.

All assertions exercise the pure-format helper
:func:`views.errors._sources_equation_markdown` so the tests run without
a Streamlit script context.
"""

from __future__ import annotations

import datetime as dt

import pytest

from orm.error_read_model import ErrorSource
from views.errors import _sources_equation_markdown


# Convenience aliases for readability inside individual assertions.
EL = ErrorSource.ERROR_LOG.value
AR = ErrorSource.AGENT_RUN.value
FS = ErrorSource.FALLBACK_SINK.value


class TestEquationShape:
    """The textual format itself."""

    def test_all_selected_no_strikethrough(self):
        md = _sources_equation_markdown(
            {EL: 32, AR: 12, FS: 3},
            [EL, AR, FS],
        )
        assert md == "Total: 47 = Error Log (32) + Agent Runs (12) + Fallback File (3)"
        assert "~~" not in md

    def test_order_is_error_log_then_agent_runs_then_fallback(self):
        # Order is canonical regardless of how the selected list is sorted —
        # matches the Source KPI card order at views/errors.py:200-221.
        md = _sources_equation_markdown(
            {FS: 1, AR: 2, EL: 4},
            [FS, AR, EL],
        )
        assert md == "Total: 7 = Error Log (4) + Agent Runs (2) + Fallback File (1)"

    def test_uses_summed_total_not_max(self):
        md = _sources_equation_markdown(
            {EL: 10, AR: 10, FS: 10},
            [EL, AR, FS],
        )
        assert md.startswith("Total: 30 =")


class TestStrikethroughOnDeselect:
    """A deselected source strikes through its term; Total is unchanged."""

    def test_single_source_deselected_only_that_term_struck(self):
        md = _sources_equation_markdown(
            {EL: 32, AR: 12, FS: 3},
            [EL, AR],  # FS deselected
        )
        # Total still 47 (unfiltered)
        assert "Total: 47" in md
        # FS struck
        assert "~~Fallback File (3)~~" in md
        # Other terms NOT struck
        assert "Error Log (32)" in md and "~~Error Log (32)~~" not in md
        assert "Agent Runs (12)" in md and "~~Agent Runs (12)~~" not in md

    def test_two_sources_deselected_both_struck(self):
        md = _sources_equation_markdown(
            {EL: 32, AR: 12, FS: 3},
            [EL],  # AR + FS deselected
        )
        assert "Total: 47" in md
        assert "~~Agent Runs (12)~~" in md
        assert "~~Fallback File (3)~~" in md
        assert "Error Log (32)" in md and "~~Error Log (32)~~" not in md

    def test_all_three_deselected_all_struck_total_unchanged(self):
        md = _sources_equation_markdown(
            {EL: 32, AR: 12, FS: 3},
            [],  # nothing selected
        )
        # Total stays at the unfiltered cross-source sum (load-bearing invariant).
        assert "Total: 47" in md
        assert "~~Error Log (32)~~" in md
        assert "~~Agent Runs (12)~~" in md
        assert "~~Fallback File (3)~~" in md


class TestEdgeCases:
    """Empty / missing / partial-failure paths."""

    def test_empty_counts_renders_zeros(self):
        md = _sources_equation_markdown({}, [EL, AR, FS])
        assert md == "Total: 0 = Error Log (0) + Agent Runs (0) + Fallback File (0)"

    def test_one_key_missing_treated_as_zero(self):
        md = _sources_equation_markdown(
            {EL: 5, FS: 2},  # AR missing
            [EL, AR, FS],
        )
        assert "Total: 7" in md
        assert "Agent Runs (0)" in md

    def test_empty_counts_and_nothing_selected(self):
        md = _sources_equation_markdown({}, [])
        # All terms struck, Total still 0 (no raising).
        assert "Total: 0" in md
        assert "~~Error Log (0)~~" in md
        assert "~~Agent Runs (0)~~" in md
        assert "~~Fallback File (0)~~" in md

    def test_unknown_keys_in_counts_are_ignored(self):
        # A bogus source value (e.g. from a stale cache key) doesn't crash
        # and isn't added to Total.
        md = _sources_equation_markdown(
            {EL: 5, AR: 3, FS: 2, "unknown_source": 99},
            [EL, AR, FS],
        )
        assert "Total: 10" in md
        assert "unknown_source" not in md


class TestCardConsistencyInvariant:
    """For each source, the equation term is struck through iff the
    corresponding Source KPI card is dimmed (``dim=True``). Both
    observe ``selected_source_values`` identically.
    """

    @pytest.mark.parametrize(
        "src",
        [EL, AR, FS],
    )
    def test_strikethrough_matches_card_dim(self, src):
        counts = {EL: 10, AR: 20, FS: 30}
        # Deselect ONLY this source.
        selected = [s for s in (EL, AR, FS) if s != src]
        md = _sources_equation_markdown(counts, selected)
        # The Source KPI card for `src` would render with dim=True
        # (label struck through) because `src not in selected`.
        # The equation must also strike through this source's term.
        label = {EL: "Error Log", AR: "Agent Runs", FS: "Fallback File"}[src]
        n = counts[src]
        assert f"~~{label} ({n})~~" in md
        # The other two sources are selected → NOT struck through.
        for other in (EL, AR, FS):
            if other == src:
                continue
            other_label = {EL: "Error Log", AR: "Agent Runs", FS: "Fallback File"}[other]
            other_n = counts[other]
            assert f"{other_label} ({other_n})" in md
            assert f"~~{other_label} ({other_n})~~" not in md


# ---------------------------------------------------------------------------
# Cross-surface invariant: equation Total == Analytics row Total Errors KPI
# ---------------------------------------------------------------------------


class TestCrossSurfaceTotalInvariant:
    """Under the same ``days_int``, the equation's Total equals
    ``_load_aggregates(days_int)["total"]`` (the Analytics row's Total
    Errors KPI). This is the structural reconciliation that Epic #161
    is built to guarantee.
    """

    def test_equation_total_equals_aggregates_total(self, tmp_path, monkeypatch):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        from orm import error_read_model
        from orm.models import Base, ErrorLog
        from utils.error_fallback_sink import (
            ErrorLoggingConfig,
            _reset_handler_cache,
        )
        from views.errors_helpers import _query_filtered

        _reset_handler_cache()

        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine, autocommit=False, autoflush=False)
        monkeypatch.setattr(error_read_model, "SessionLocal", Session)

        # Seed three ErrorLog rows of mixed severities so query_aggregates
        # has a non-trivial Total.
        with Session() as session:
            for sev in ("warning", "error", "critical"):
                session.add(
                    ErrorLog(
                        category="sql_execution",
                        severity=sev,
                        error_type="RuntimeError",
                        error_message=f"{sev}-msg",
                        created_at=dt.datetime.now() - dt.timedelta(hours=1),
                    )
                )
            session.commit()

        cfg = ErrorLoggingConfig(
            fallback_path=tmp_path / "x.jsonl",
            fallback_max_bytes=5_000_000,
            fallback_backup_count=5,
        )

        # Counts as the view sees them (unfiltered across sources).
        _, counts = _query_filtered(
            days=7,
            sources_csv="",  # all three
            categories_csv="",
            severities_csv="",
            search="",
            user_id_text="",
            fallback_config=cfg,
        )

        # Aggregates as the Analytics row sees them.
        agg = error_read_model.query_aggregates(
            dt.datetime.now() - dt.timedelta(days=7),
            fallback_config=cfg,
        )

        # The equation's Total under any selected-source state equals
        # the unfiltered cross-source sum, which by construction equals
        # aggregates.total.
        equation_md = _sources_equation_markdown(counts, [EL, AR, FS])
        assert f"Total: {agg.total}" in equation_md, (
            f"Equation Total must equal aggregates.total ({agg.total}); got equation: {equation_md!r}"
        )

        # And it stays equal even when a source is deselected
        # (filter state is signaled via strike-through, not by recalculating).
        equation_md_partial = _sources_equation_markdown(counts, [EL])
        assert f"Total: {agg.total}" in equation_md_partial, (
            "Equation Total must remain at the unfiltered sum even when "
            "Source chips are deselected — otherwise Sources Total drifts "
            "from Analytics Total, the exact drift Epic #161 prevents."
        )


# ---------------------------------------------------------------------------
# Call-site wiring — the wrapper hands the equation to st.markdown
# ---------------------------------------------------------------------------


class TestRenderWrapperDelegates:
    """The thin Streamlit wrapper renders the pure-format string."""

    def test_render_sources_equation_calls_st_markdown(self, monkeypatch):
        from views import errors

        markdown_calls: list[str] = []

        class _StubSt:
            def markdown(self, body, **_kw):
                markdown_calls.append(str(body))

        monkeypatch.setattr(errors, "st", _StubSt())

        errors._render_sources_equation({EL: 1, AR: 2, FS: 3}, [EL, AR, FS])
        assert markdown_calls == ["Total: 6 = Error Log (1) + Agent Runs (2) + Fallback File (3)"]
