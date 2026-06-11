"""Tests for the Admin Actions audit tab row-click dialog (#156).

Covers:
  (a) Dialog opens when row selection state changes; closes when cleared.
  (b) Both old/new JSON columns render when both present.
  (c) Only the populated column renders when one is null.
  (d) Malformed JSON falls back to ``st.text`` without raising.
  (e) Error block renders only when ``success=False`` and message present.
  (f) Affected-count line renders only when non-null.
  (g) Target line formatting prefers username, falls back to entity_type:entity_id.
  (h) Pre-existing KPI / Action Distribution surface stays unchanged
      (the per-row expander is gone — dialog is the sole detail surface).
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_action(
    *,
    id: int = 1,
    admin_username: str = "alice_admin",
    action_type: str = "user_update",
    target_user_id: int | None = 42,
    target_username: str | None = "bob",
    target_entity_type: str | None = "user",
    target_entity_id: str | None = "42",
    description: str = "Updated bob's role",
    old_value: str | None = '{"role": "Patient"}',
    new_value: str | None = '{"role": "Doctor"}',
    affected_count: int | None = 1,
    success: bool | None = True,
    error_message: str | None = None,
    created_at: datetime | None = None,
) -> dict:
    return {
        "id": id,
        "created_at": created_at or datetime(2026, 6, 1, 12, 0, 0),
        "admin_username": admin_username,
        "action_type": action_type,
        "target_user_id": target_user_id,
        "target_username": target_username,
        "target_entity_type": target_entity_type,
        "target_entity_id": target_entity_id,
        "description": description,
        "old_value": old_value,
        "new_value": new_value,
        "affected_count": affected_count,
        "success": success,
        "error_message": error_message,
    }


class _RecordingStreamlit:
    """A minimal recording stand-in for the ``st`` module for the dialog body.

    Mirrors the pattern from tests/views/test_admin_audit_dialog.py (#155).
    Only the surface used by ``_render_admin_action_dialog_body`` is
    implemented.
    """

    def __init__(self):
        self.session_state: dict = {}
        self.code_calls: list[tuple[str, str]] = []  # (content, language)
        self.json_calls: list = []
        self.write_calls: list = []
        self.text_calls: list = []
        self.markdown_calls: list[str] = []
        self.caption_calls: list[str] = []
        self.columns_calls: list[int] = []

    # ---- Streamlit surface ------------------------------------------------
    def markdown(self, body, **_kwargs):
        self.markdown_calls.append(str(body))

    def write(self, body, **_kwargs):
        self.write_calls.append(body)

    def code(self, body, language: str = "text", **_kwargs):
        self.code_calls.append((str(body), language))

    def json(self, body, **_kwargs):
        self.json_calls.append(body)

    def text(self, body, **_kwargs):
        self.text_calls.append(str(body))

    def caption(self, body, **_kwargs):
        self.caption_calls.append(str(body))

    def divider(self):
        pass

    def columns(self, spec):
        n = spec if isinstance(spec, int) else len(spec)
        self.columns_calls.append(n)
        return [MagicMock(__enter__=lambda s: s, __exit__=lambda *a: None) for _ in range(n)]

    def dialog(self, _title):
        def _decorator(fn):
            return fn

        return _decorator


# ---------------------------------------------------------------------------
# (b)(c)(d) Old / New JSON columns
# ---------------------------------------------------------------------------


class TestOldNewJsonColumns:
    def test_both_columns_render_when_both_present(self):
        from views import admin_analytics

        rec = _RecordingStreamlit()
        item = _make_action(
            old_value='{"role": "Patient"}',
            new_value='{"role": "Doctor"}',
        )
        with patch.object(admin_analytics, "st", rec):
            admin_analytics._render_admin_action_dialog_body(item)

        assert len(rec.json_calls) == 2, "Both old AND new JSON blocks must render when both are present"
        # Order: old then new.
        assert rec.json_calls[0] == {"role": "Patient"}
        assert rec.json_calls[1] == {"role": "Doctor"}
        # st.text fallback must NOT have been invoked for either column.
        assert rec.text_calls == []

    def test_only_old_renders_when_new_is_null(self):
        from views import admin_analytics

        rec = _RecordingStreamlit()
        item = _make_action(old_value='{"k": 1}', new_value=None)
        with patch.object(admin_analytics, "st", rec):
            admin_analytics._render_admin_action_dialog_body(item)

        assert rec.json_calls == [{"k": 1}], "Only the populated (old) JSON column should render"

    def test_only_new_renders_when_old_is_null(self):
        from views import admin_analytics

        rec = _RecordingStreamlit()
        item = _make_action(old_value=None, new_value='{"k": 2}')
        with patch.object(admin_analytics, "st", rec):
            admin_analytics._render_admin_action_dialog_body(item)

        assert rec.json_calls == [{"k": 2}], "Only the populated (new) JSON column should render"

    def test_empty_string_is_treated_as_absent(self):
        """``""`` should be treated the same as ``None`` — neither column
        renders if both are empty strings."""
        from views import admin_analytics

        rec = _RecordingStreamlit()
        item = _make_action(old_value="", new_value="")
        with patch.object(admin_analytics, "st", rec):
            admin_analytics._render_admin_action_dialog_body(item)

        assert rec.json_calls == []
        assert rec.text_calls == []

    def test_malformed_old_value_falls_back_to_text(self):
        from views import admin_analytics

        rec = _RecordingStreamlit()
        item = _make_action(old_value="not-a-json {{{", new_value=None)
        with patch.object(admin_analytics, "st", rec):
            # Must not raise.
            admin_analytics._render_admin_action_dialog_body(item)

        assert rec.json_calls == []
        assert "not-a-json" in " ".join(rec.text_calls)

    def test_malformed_new_value_falls_back_to_text(self):
        from views import admin_analytics

        rec = _RecordingStreamlit()
        item = _make_action(old_value=None, new_value="<<garbage>>")
        with patch.object(admin_analytics, "st", rec):
            admin_analytics._render_admin_action_dialog_body(item)

        assert rec.json_calls == []
        assert "<<garbage>>" in " ".join(rec.text_calls)

    def test_one_malformed_one_valid_each_handled_independently(self):
        from views import admin_analytics

        rec = _RecordingStreamlit()
        item = _make_action(old_value="not json", new_value='{"ok": true}')
        with patch.object(admin_analytics, "st", rec):
            admin_analytics._render_admin_action_dialog_body(item)

        assert rec.json_calls == [{"ok": True}]
        assert any("not json" in t for t in rec.text_calls)


# ---------------------------------------------------------------------------
# (e) Error block — gated on (success=False AND message present)
# ---------------------------------------------------------------------------


class TestErrorBlock:
    def test_error_block_renders_when_success_false_and_message_present(self):
        from views import admin_analytics

        rec = _RecordingStreamlit()
        item = _make_action(success=False, error_message="DB constraint violated")
        with patch.object(admin_analytics, "st", rec):
            admin_analytics._render_admin_action_dialog_body(item)

        error_codes = [body for body, lang in rec.code_calls if lang == "text"]
        assert any("DB constraint violated" in body for body in error_codes), (
            "Expected the error_message to be rendered as a text code block"
        )

    def test_no_error_block_when_success_true_even_with_message(self):
        """If success=True, the error_message field is meaningless and must
        not render."""
        from views import admin_analytics

        rec = _RecordingStreamlit()
        item = _make_action(success=True, error_message="stale warning")
        with patch.object(admin_analytics, "st", rec):
            admin_analytics._render_admin_action_dialog_body(item)

        error_codes = [body for body, lang in rec.code_calls if lang == "text"]
        assert error_codes == [], "Error block must not render when success=True"

    def test_no_error_block_when_failed_but_no_message(self):
        from views import admin_analytics

        rec = _RecordingStreamlit()
        item = _make_action(success=False, error_message=None)
        with patch.object(admin_analytics, "st", rec):
            admin_analytics._render_admin_action_dialog_body(item)

        error_codes = [body for body, lang in rec.code_calls if lang == "text"]
        assert error_codes == [], "Error block requires a non-empty error_message"

    def test_failed_badge_renders_when_success_false(self):
        from views import admin_analytics

        rec = _RecordingStreamlit()
        item = _make_action(success=False, error_message=None)
        with patch.object(admin_analytics, "st", rec):
            admin_analytics._render_admin_action_dialog_body(item)

        badges = " ".join(rec.markdown_calls)
        assert "FAILED" in badges and "SUCCESS" not in badges.replace("FAILED", "")

    def test_success_badge_renders_when_success_true(self):
        from views import admin_analytics

        rec = _RecordingStreamlit()
        item = _make_action(success=True)
        with patch.object(admin_analytics, "st", rec):
            admin_analytics._render_admin_action_dialog_body(item)

        assert any("SUCCESS" in m for m in rec.markdown_calls)


# ---------------------------------------------------------------------------
# (f) Affected-count line
# ---------------------------------------------------------------------------


class TestAffectedCount:
    def test_affected_renders_when_non_null(self):
        from views import admin_analytics

        rec = _RecordingStreamlit()
        item = _make_action(affected_count=37)
        with patch.object(admin_analytics, "st", rec):
            admin_analytics._render_admin_action_dialog_body(item)

        assert any("Affected" in m and "37" in m for m in rec.markdown_calls)

    def test_affected_omitted_when_null(self):
        from views import admin_analytics

        rec = _RecordingStreamlit()
        item = _make_action(affected_count=None)
        with patch.object(admin_analytics, "st", rec):
            admin_analytics._render_admin_action_dialog_body(item)

        assert not any("Affected" in m for m in rec.markdown_calls)

    def test_affected_zero_is_treated_as_present(self):
        """Zero is a meaningful value for bulk operations — don't drop it."""
        from views import admin_analytics

        rec = _RecordingStreamlit()
        item = _make_action(affected_count=0)
        with patch.object(admin_analytics, "st", rec):
            admin_analytics._render_admin_action_dialog_body(item)

        assert any("Affected" in m and "0" in m for m in rec.markdown_calls)


# ---------------------------------------------------------------------------
# (g) Target line formatting
# ---------------------------------------------------------------------------


class TestTargetLine:
    def test_username_preferred_when_present(self):
        from views import admin_analytics

        rec = _RecordingStreamlit()
        item = _make_action(target_username="bob", target_entity_type="user", target_entity_id="42")
        with patch.object(admin_analytics, "st", rec):
            admin_analytics._render_admin_action_dialog_body(item)

        target_md = [m for m in rec.markdown_calls if m.startswith("**Target:**")]
        assert target_md == ["**Target:** bob"]

    def test_falls_back_to_entity_type_id_when_no_username(self):
        from views import admin_analytics

        rec = _RecordingStreamlit()
        item = _make_action(target_username=None, target_entity_type="training_data", target_entity_id="abc-123")
        with patch.object(admin_analytics, "st", rec):
            admin_analytics._render_admin_action_dialog_body(item)

        target_md = [m for m in rec.markdown_calls if m.startswith("**Target:**")]
        assert target_md == ["**Target:** training_data:abc-123"]

    def test_target_line_omitted_when_no_target_info(self):
        from views import admin_analytics

        rec = _RecordingStreamlit()
        item = _make_action(
            target_user_id=None,
            target_username=None,
            target_entity_type=None,
            target_entity_id=None,
        )
        with patch.object(admin_analytics, "st", rec):
            admin_analytics._render_admin_action_dialog_body(item)

        assert not any(m.startswith("**Target:**") for m in rec.markdown_calls)


# ---------------------------------------------------------------------------
# (a) Selection state opens / closes the dialog
# ---------------------------------------------------------------------------


def _make_tab_stub():
    """Build a stub ``st`` module sufficient for ``_render_audit_tab`` to run."""

    class _Stub:
        def __init__(self):
            self.session_state = {}
            self.captured_dataframe_kwargs: list[dict] = []
            self.dialog_invocations: list[dict] = []
            # Feature #170: ``st.column_config.Column(...)`` is now invoked
            # by ``_render_audit_tab`` when wiring the leading View column.
            self.column_config = MagicMock()

        # Streamlit surface --------------------------------------------------
        def columns(self, spec):
            n = spec if isinstance(spec, int) else len(spec)
            return [MagicMock(__enter__=lambda s: s, __exit__=lambda *a: None) for _ in range(n)]

        def container(self, **_kw):
            return MagicMock(__enter__=lambda s: s, __exit__=lambda *a: None)

        def divider(self):
            pass

        def subheader(self, *_a, **_kw):
            pass

        def info(self, *_a, **_kw):
            pass

        def selectbox(self, *_a, options=None, index=0, key=None, **_kw):
            opts = list(options or [])
            val = opts[index] if opts else None
            if key:
                self.session_state.setdefault(key, val)
            return val

        def number_input(self, *_a, key=None, **_kw):
            self.session_state.setdefault(key, 1)
            return 1

        def dataframe(self, _df, **kwargs):
            self.captured_dataframe_kwargs.append(kwargs)
            # Default: simulate the configured selection_event hook (set per test).
            return getattr(self, "_dataframe_event", None) or MagicMock(selection={"rows": []})

        def button(self, *_a, **_kw):
            return False

        def caption(self, *_a, **_kw):
            pass

        def markdown(self, *_a, **_kw):
            pass

        def write(self, *_a, **_kw):
            pass

        def code(self, *_a, **_kw):
            pass

        def text(self, *_a, **_kw):
            pass

        def json(self, *_a, **_kw):
            pass

        def plotly_chart(self, *_a, **_kw):
            pass

        def warning(self, *_a, **_kw):
            pass

        def error(self, *_a, **_kw):
            pass

        def download_button(self, *_a, **_kw):
            pass

        def dialog(self, _title):
            def _decorator(fn):
                return fn

            return _decorator

    return _Stub()


class TestSelectionStateTrigger:
    def test_row_selection_opens_dialog_and_sets_state(self):
        """When the dataframe event reports a selected row, _render_audit_tab
        must set ``audit_actions_dialog_open_id`` and invoke the dialog
        function for that row."""
        from views import admin_analytics

        item = _make_action(id=314)
        dialog_calls: list[dict] = []

        stub = _make_tab_stub()
        ev = MagicMock()
        ev.selection = {"rows": [0]}
        stub._dataframe_event = ev

        with (
            patch.object(admin_analytics, "st", stub),
            patch.object(admin_analytics, "_render_admin_action_dialog", side_effect=lambda x: dialog_calls.append(x)),
            patch(
                "orm.logging_functions.get_admin_actions_page",
                return_value={"items": [item], "total": 1},
            ),
            patch(
                "orm.logging_functions.get_admin_action_stats",
                return_value={"total": 1, "user_changes": 0, "training_actions": 0, "failed": 0},
            ),
            patch("orm.logging_functions.get_admin_actions_by_type", return_value=[]),
        ):
            admin_analytics._render_audit_tab(30)

        assert dialog_calls == [item], "Dialog must be invoked for the selected row's item"
        assert stub.session_state.get("audit_actions_dialog_open_id") == 314

    def test_no_selection_clears_dialog_state_key(self):
        from views import admin_analytics

        item = _make_action(id=314)

        stub = _make_tab_stub()
        # Pre-existing tracking key — must be cleared when selection is empty.
        stub.session_state["audit_actions_dialog_open_id"] = 314
        ev = MagicMock()
        ev.selection = {"rows": []}
        stub._dataframe_event = ev

        with (
            patch.object(admin_analytics, "st", stub),
            patch.object(admin_analytics, "_render_admin_action_dialog") as dialog_mock,
            patch(
                "orm.logging_functions.get_admin_actions_page",
                return_value={"items": [item], "total": 1},
            ),
            patch(
                "orm.logging_functions.get_admin_action_stats",
                return_value={"total": 1, "user_changes": 0, "training_actions": 0, "failed": 0},
            ),
            patch("orm.logging_functions.get_admin_actions_by_type", return_value=[]),
        ):
            admin_analytics._render_audit_tab(30)

        dialog_mock.assert_not_called()
        assert "audit_actions_dialog_open_id" not in stub.session_state

    def test_dataframe_wired_with_single_row_selection_primitive(self):
        """The grid must always use the locked-in trigger primitive:
        ``selection_mode='single-row'``, ``on_select='rerun'``, and a stable
        key. Epic #154 mandates this for #156 + #157."""
        from views import admin_analytics

        item = _make_action()
        stub = _make_tab_stub()

        with (
            patch.object(admin_analytics, "st", stub),
            patch.object(admin_analytics, "_render_admin_action_dialog"),
            patch(
                "orm.logging_functions.get_admin_actions_page",
                return_value={"items": [item], "total": 1},
            ),
            patch(
                "orm.logging_functions.get_admin_action_stats",
                return_value={"total": 1, "user_changes": 0, "training_actions": 0, "failed": 0},
            ),
            patch("orm.logging_functions.get_admin_actions_by_type", return_value=[]),
        ):
            admin_analytics._render_audit_tab(30)

        assert stub.captured_dataframe_kwargs, "Expected st.dataframe to be called for the grid"
        kw = stub.captured_dataframe_kwargs[0]
        assert kw.get("selection_mode") == "single-row"
        assert kw.get("on_select") == "rerun"
        assert kw.get("key") == "audit_actions_dataframe"


# ---------------------------------------------------------------------------
# Per-row expander removal & no get_recent_admin_actions reference
# ---------------------------------------------------------------------------


class TestExpanderRetired:
    def test_admin_analytics_no_longer_imports_get_recent_admin_actions(self):
        """The per-row expander loop is gone — `get_recent_admin_actions`
        must not appear anywhere in admin_analytics.py."""
        from pathlib import Path

        src = Path(__file__).resolve().parents[2] / "views" / "admin_analytics.py"
        text = src.read_text()
        assert "get_recent_admin_actions" not in text, (
            "Sole caller removed in #156; the import + call must be gone from views/admin_analytics.py"
        )

    def test_get_recent_admin_actions_is_removed_from_logging_functions(self):
        """Backend function itself is retired (the spec calls for removal,
        not deprecation, since it had a single call site)."""
        from orm import logging_functions

        assert not hasattr(logging_functions, "get_recent_admin_actions"), (
            "get_recent_admin_actions should be removed; the dialog page reads via get_admin_actions_page"
        )


# ---------------------------------------------------------------------------
# Pre-existing KPI / Action Distribution surface untouched
# ---------------------------------------------------------------------------


class TestKpiAndDistributionPreserved:
    def test_kpi_cards_and_distribution_still_render_above_grid(self):
        """The KPI row + distribution pie/table block should still call
        ``get_admin_action_stats`` and ``get_admin_actions_by_type`` —
        these were not touched by #156."""
        from views import admin_analytics

        stats_calls: list = []
        by_type_calls: list = []

        def _stats(days=30):
            stats_calls.append(days)
            return {"total": 5, "user_changes": 2, "training_actions": 1, "failed": 1}

        def _by_type(days=30):
            by_type_calls.append(days)
            return [{"action_type": "user_update", "count": 3}, {"action_type": "user_delete", "count": 2}]

        stub = _make_tab_stub()

        with (
            patch.object(admin_analytics, "st", stub),
            patch.object(admin_analytics, "_render_admin_action_dialog"),
            patch("orm.logging_functions.get_admin_action_stats", side_effect=_stats),
            patch("orm.logging_functions.get_admin_actions_by_type", side_effect=_by_type),
            patch(
                "orm.logging_functions.get_admin_actions_page",
                return_value={"items": [], "total": 0},
            ),
        ):
            admin_analytics._render_audit_tab(7)

        assert stats_calls == [7]
        assert by_type_calls == [7]


# ---------------------------------------------------------------------------
# Sanity: dialog body doesn't raise on minimal item
# ---------------------------------------------------------------------------


def test_minimal_item_does_not_raise():
    """A row with nothing but id + admin_username + action_type should still
    render cleanly (no Target line, no JSON diff, no error, no affected)."""
    from views import admin_analytics

    rec = _RecordingStreamlit()
    item = {
        "id": 1,
        "created_at": None,
        "admin_username": None,
        "action_type": None,
        "target_user_id": None,
        "target_username": None,
        "target_entity_type": None,
        "target_entity_id": None,
        "description": None,
        "old_value": None,
        "new_value": None,
        "affected_count": None,
        "success": None,
        "error_message": None,
    }
    with patch.object(admin_analytics, "st", rec):
        admin_analytics._render_admin_action_dialog_body(item)

    # Caption with Action ID 1 should always appear at the end.
    assert any("Action ID 1" in c for c in rec.caption_calls)
