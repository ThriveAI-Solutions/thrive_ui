"""Tests for Epic #169 / Feature #170 — the labeled ``View`` checkbox
column + ``View Selected`` action button shared by all three audit
tables (Questions, Admin Actions, User Activity).

Covers, parametrised across the 3 tabs where possible:

  - The ``View`` column is configured as ``st.column_config.CheckboxColumn``
    with the spec's label and help tooltip.
  - All non-View data columns are configured with ``disabled=True``.
  - The ``View Selected`` button is disabled when 0 or >1 rows are
    checked, enabled with exactly 1; its tooltip when disabled equals
    ``_VIEW_SELECTED_DISABLED_HELP``.
  - Clicking the button with exactly one row checked invokes the
    corresponding detail dialog with the matching row payload.
  - CSV export for the Questions tab does NOT include the ``View``
    column (it lives only in the on-screen ``edited_df``).
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixture builders — one row matching each tab's loader shape
# ---------------------------------------------------------------------------


def _make_question_item(*, user_message_id: int = 1) -> dict:
    return {
        "asked_at": datetime(2026, 6, 1, 12, 0, 0),
        "user_id": 7,
        "username": "alice",
        "organization": "Acme",
        "question": "How many patients today?",
        "sql_text": "SELECT count(*) FROM patient",
        "status": "Success",
        "elapsed_seconds": 1.0,
        "summary_text": "12 patients.",
        "dataframe_preview": None,
        "error_text": None,
        "user_message_id": user_message_id,
        "scope": "Patient",
    }


def _make_admin_action_item(*, id: int = 1) -> dict:
    return {
        "id": id,
        "created_at": datetime(2026, 6, 1, 12, 0, 0),
        "admin_username": "alice_admin",
        "action_type": "user_update",
        "target_user_id": 42,
        "target_username": "bob",
        "target_entity_type": "user",
        "target_entity_id": "42",
        "description": "Updated bob's role",
        "old_value": '{"role": "Patient"}',
        "new_value": '{"role": "Doctor"}',
        "affected_count": 1,
        "success": True,
        "error_message": None,
    }


def _make_user_activity_item(*, id: int = 1) -> dict:
    return {
        "id": id,
        "created_at": datetime(2026, 6, 1, 12, 0, 0),
        "user_id": 9,
        "username": "carol",
        "activity_type": "settings_change",
        "description": "Changed theme",
        "old_value": None,
        "new_value": None,
        "ip_address": "127.0.0.1",
        "user_agent": "Mozilla/5.0 Test",
    }


# ---------------------------------------------------------------------------
# Per-tab harness — wraps the three tab functions behind one interface so
# the parametrised tests stay readable.
# ---------------------------------------------------------------------------


class _TabHarness:
    """Encapsulates per-tab details (render function, item factory,
    data_editor key, View Selected button key, dialog function name)
    so the parametrised tests can drive any of the 3 audit tabs from
    a single body."""

    def __init__(
        self,
        *,
        label: str,
        render_fn_name: str,
        item_factory,
        data_editor_key: str,
        button_key: str,
        dialog_attr: str,
        loader_patches: list,
    ):
        self.label = label
        self.render_fn_name = render_fn_name
        self.item_factory = item_factory
        self.data_editor_key = data_editor_key
        self.button_key = button_key
        self.dialog_attr = dialog_attr
        self.loader_patches = loader_patches  # callable returning list of patchers

    def __repr__(self) -> str:  # nicer pytest IDs
        return self.label


def _questions_loader_patches(items):
    from views import admin_analytics

    return [
        patch.object(
            admin_analytics,
            "_cached_audit_filter_options",
            return_value={"usernames": [], "orgs": []},
        ),
        patch(
            "orm.logging_functions.get_question_audit_page",
            return_value={"items": list(items), "total": len(items)},
        ),
        patch("orm.functions.get_all_users", return_value=[]),
    ]


def _actions_loader_patches(items):
    return [
        patch(
            "orm.logging_functions.get_admin_actions_page",
            return_value={"items": list(items), "total": len(items)},
        ),
        patch(
            "orm.logging_functions.get_admin_action_stats",
            return_value={"total": 1, "user_changes": 0, "training_actions": 0, "failed": 0},
        ),
        patch("orm.logging_functions.get_admin_actions_by_type", return_value=[]),
    ]


def _activity_loader_patches(items):
    return [
        patch(
            "orm.logging_functions.get_activity_stats",
            return_value={
                "logins_today": 0,
                "failed_logins": 0,
                "settings_changes": 0,
                "unique_users": 0,
            },
        ),
        patch("orm.logging_functions.get_activity_over_time", return_value=[]),
        patch("orm.logging_functions.get_activity_by_type", return_value=[]),
        patch(
            "orm.logging_functions.get_user_activity_page",
            return_value={"items": list(items), "total": len(items)},
        ),
    ]


def _all_harnesses() -> list[_TabHarness]:
    return [
        _TabHarness(
            label="questions",
            render_fn_name="_render_audit_trail_tab",
            item_factory=lambda i: _make_question_item(user_message_id=i),
            data_editor_key="audit_dataframe",
            button_key="audit_questions_view_selected_btn",
            dialog_attr="_render_audit_question_dialog",
            loader_patches=_questions_loader_patches,
        ),
        _TabHarness(
            label="admin_actions",
            render_fn_name="_render_audit_tab",
            item_factory=lambda i: _make_admin_action_item(id=i),
            data_editor_key="audit_actions_dataframe",
            button_key="audit_actions_view_selected_btn",
            dialog_attr="_render_admin_action_dialog",
            loader_patches=_actions_loader_patches,
        ),
        _TabHarness(
            label="user_activity",
            render_fn_name="_render_activity_tab",
            item_factory=lambda i: _make_user_activity_item(id=i),
            data_editor_key="audit_activity_dataframe",
            button_key="audit_activity_view_selected_btn",
            dialog_attr="_render_user_activity_dialog",
            loader_patches=_activity_loader_patches,
        ),
    ]


# ---------------------------------------------------------------------------
# Streamlit stub that records ``column_config`` args (CheckboxColumn /
# TextColumn) so tests can introspect what the tab passed.
# ---------------------------------------------------------------------------


class _ColumnConfigCall:
    """A captured ``st.column_config.<Kind>(label, **kwargs)`` call.
    Mimics enough surface to flow through ``column_config={...}`` to
    ``st.data_editor`` and back so tests can introspect it."""

    def __init__(self, kind: str, args: tuple, kwargs: dict):
        self.kind = kind
        self.args = args
        self.kwargs = kwargs

    @property
    def label(self):
        if self.args:
            return self.args[0]
        return self.kwargs.get("label")

    @property
    def help(self):
        return self.kwargs.get("help")

    @property
    def disabled(self) -> bool:
        return bool(self.kwargs.get("disabled", False))


class _ColumnConfig:
    """Stand-in for ``st.column_config`` that records each
    ``CheckboxColumn(...)`` / ``TextColumn(...)`` call."""

    def __init__(self):
        self.calls: list[_ColumnConfigCall] = []

    def CheckboxColumn(self, *args, **kwargs):  # noqa: N802 — mirror Streamlit's name
        call = _ColumnConfigCall("CheckboxColumn", args, kwargs)
        self.calls.append(call)
        return call

    def TextColumn(self, *args, **kwargs):  # noqa: N802
        call = _ColumnConfigCall("TextColumn", args, kwargs)
        self.calls.append(call)
        return call


def _make_stub(
    *,
    view_column_values: list[bool] | None = None,
    button_clicks: dict | None = None,
    capture_button_kwargs: dict | None = None,
):
    """Build a streamlit stub for any of the three tab render functions.

    ``view_column_values`` is the list of per-row checkbox values written
    back into ``edited_df`` (simulating user toggles).
    ``button_clicks`` is a dict keyed by button ``key=`` value; lets
    tests fire any specific button on demand.
    ``capture_button_kwargs`` (optional output dict): if passed, the
    stub appends ``{label, key, kwargs}`` for every ``st.button(...)``
    call into ``capture_button_kwargs["calls"]``.
    """

    button_clicks = button_clicks or {}

    class _Stub:
        def __init__(self):
            self.session_state = {}
            self.secrets = {"agent_logging": {"mode": "full"}}
            self.components = MagicMock()
            self.components.v1 = MagicMock()
            self.components.v1.html = MagicMock()
            self.column_config = _ColumnConfig()
            self.captured_data_editor_kwargs: list[dict] = []

        # -- Layout / widgets ----------------------------------------------
        def multiselect(self, *_a, key=None, **_kw):
            if key is not None:
                self.session_state.setdefault(key, [])
            return []

        def text_input(self, *_a, key=None, **_kw):
            if key is not None:
                self.session_state.setdefault(key, "")
            return ""

        def selectbox(self, *_a, options=None, index=0, key=None, **_kw):
            opts = list(options or [])
            val = opts[index] if opts else None
            if key:
                self.session_state.setdefault(key, val)
            return val

        def number_input(self, *_a, key=None, **_kw):
            if key is not None:
                self.session_state.setdefault(key, 1)
            return 1

        def columns(self, spec, **_kw):
            n = spec if isinstance(spec, int) else len(spec)
            return [MagicMock(__enter__=lambda s: s, __exit__=lambda *a: None) for _ in range(n)]

        def container(self, **_kw):
            return MagicMock(__enter__=lambda s: s, __exit__=lambda *a: None)

        def metric(self, *_a, **_kw):
            pass

        def divider(self):
            pass

        def subheader(self, *_a, **_kw):
            pass

        def info(self, *_a, **_kw):
            pass

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

        # -- The two grid primitives ---------------------------------------
        def data_editor(self, df, **kwargs):
            self.captured_data_editor_kwargs.append(kwargs)
            out = df.copy()
            if "View" in out.columns and view_column_values:
                vals = list(view_column_values) + [False] * max(0, len(out) - len(view_column_values))
                out["View"] = vals[: len(out)]
            return out

        def dataframe(self, _df, **_kwargs):
            return MagicMock()

        # -- Buttons -------------------------------------------------------
        def button(self, label="", key=None, **kwargs):
            if capture_button_kwargs is not None:
                capture_button_kwargs.setdefault("calls", []).append({"label": label, "key": key, "kwargs": kwargs})
            return bool(button_clicks.get(key, False))

        def download_button(self, *_a, **_kw):
            pass

        # -- Dialog decorator passthrough ----------------------------------
        def dialog(self, _title):
            def _decorator(fn):
                return fn

            return _decorator

    return _Stub()


def _run_tab(harness: _TabHarness, stub, item, dialog_recorder=None):
    """Drive any tab's render function with the given stub, item, and
    (optional) dialog spy. Centralises the common boilerplate."""
    from views import admin_analytics

    import contextlib

    dialog_target = getattr(admin_analytics, harness.dialog_attr)
    spy_kwargs = {}
    if dialog_recorder is not None:
        spy_kwargs["side_effect"] = dialog_recorder

    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(admin_analytics, "st", stub))
        stack.enter_context(patch.object(admin_analytics, harness.dialog_attr, **spy_kwargs))
        for p in harness.loader_patches([item]):
            stack.enter_context(p)
        getattr(admin_analytics, harness.render_fn_name)(30)
    return dialog_target


# ---------------------------------------------------------------------------
# Test 1 — View column is a CheckboxColumn with the spec label + help
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("harness", _all_harnesses(), ids=lambda h: h.label)
def test_view_column_is_checkbox_column_with_label_and_help(harness):
    """Every audit tab's data_editor must configure the leading ``View``
    column as ``st.column_config.CheckboxColumn`` with ``label='View'``
    and the spec's ``help`` tooltip."""
    from views import admin_analytics

    stub = _make_stub(view_column_values=[])
    _run_tab(harness, stub, harness.item_factory(1))

    assert stub.captured_data_editor_kwargs, f"Expected st.data_editor to be called on {harness.label}"
    cc = stub.captured_data_editor_kwargs[0].get("column_config")
    assert cc and "View" in cc, (
        f"{harness.label}: column_config must include a 'View' entry; saw keys={list((cc or {}).keys())}"
    )

    view_call: _ColumnConfigCall = cc["View"]
    assert view_call.kind == "CheckboxColumn", (
        f"{harness.label}: View column must be a CheckboxColumn; saw {view_call.kind}"
    )
    assert view_call.label == "View", f"{harness.label}: View column label must equal 'View'"
    assert view_call.help == admin_analytics._VIEW_COLUMN_HELP, (
        f"{harness.label}: View column help must equal _VIEW_COLUMN_HELP"
    )


# ---------------------------------------------------------------------------
# Test 2 — All non-View columns are disabled
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("harness", _all_harnesses(), ids=lambda h: h.label)
def test_non_view_columns_are_disabled(harness):
    """Every non-View entry in the data_editor's ``column_config`` must
    carry ``disabled=True`` so admins can't accidentally edit audit data."""
    stub = _make_stub(view_column_values=[])
    _run_tab(harness, stub, harness.item_factory(1))

    cc = stub.captured_data_editor_kwargs[0].get("column_config")
    assert cc, f"{harness.label}: column_config must be passed to st.data_editor"
    for col_name, call in cc.items():
        if col_name == "View":
            continue
        assert call.disabled is True, f"{harness.label}: non-View column {col_name!r} must be disabled=True"


# ---------------------------------------------------------------------------
# Test 3 — View Selected button enabled state (0 / 1 / many)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("harness", _all_harnesses(), ids=lambda h: h.label)
@pytest.mark.parametrize(
    "checked_count,expected_disabled",
    [(0, True), (1, False), (2, True), (3, True)],
    ids=["zero", "one", "two", "three"],
)
def test_view_selected_button_enabled_state(harness, checked_count, expected_disabled):
    """``View Selected`` button is disabled when 0 or >1 rows are
    checked; enabled when exactly 1 is checked. When disabled, the
    button's ``help`` tooltip equals ``_VIEW_SELECTED_DISABLED_HELP``."""
    from views import admin_analytics

    # We need ``len(items)`` to equal ``checked_count`` worth of rows so
    # the dataframe slot exists for each checkbox value. Pre-build enough
    # items.
    items = [harness.item_factory(i) for i in range(1, max(1, checked_count) + 1)]
    view_vals = [True] * checked_count + [False] * (len(items) - checked_count)

    button_kwargs_capture: dict = {}
    stub = _make_stub(view_column_values=view_vals, capture_button_kwargs=button_kwargs_capture)

    # Patch loader to return our pre-built items list, NOT just one row.
    def _swap_loaders(_items):
        return harness.loader_patches(items)

    swapped = _TabHarness(
        label=harness.label,
        render_fn_name=harness.render_fn_name,
        item_factory=harness.item_factory,
        data_editor_key=harness.data_editor_key,
        button_key=harness.button_key,
        dialog_attr=harness.dialog_attr,
        loader_patches=_swap_loaders,
    )

    _run_tab(swapped, stub, items[0])

    # Find the View Selected button call by its key.
    btn_calls = [c for c in button_kwargs_capture.get("calls", []) if c["key"] == harness.button_key]
    assert btn_calls, (
        f"{harness.label}: ``st.button({harness.button_key!r})`` was never called — "
        "the View Selected button must always render below the data_editor"
    )
    call = btn_calls[0]
    assert call["kwargs"].get("disabled") is expected_disabled, (
        f"{harness.label} / checked={checked_count}: button disabled state mismatch; "
        f"got disabled={call['kwargs'].get('disabled')}, expected {expected_disabled}"
    )
    if expected_disabled:
        assert call["kwargs"].get("help") == admin_analytics._VIEW_SELECTED_DISABLED_HELP, (
            f"{harness.label}: disabled-state tooltip must equal _VIEW_SELECTED_DISABLED_HELP"
        )
    # Label always equals _VIEW_SELECTED_BUTTON_LABEL.
    assert call["label"] == admin_analytics._VIEW_SELECTED_BUTTON_LABEL


# ---------------------------------------------------------------------------
# Test 4 — Button click with one row checked fires the dialog
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("harness", _all_harnesses(), ids=lambda h: h.label)
def test_button_click_with_one_row_checked_invokes_dialog(harness):
    """With exactly one ``View`` checkbox ticked and the View Selected
    button clicked, the corresponding dialog function is invoked with
    the matching row's payload."""
    item = harness.item_factory(314)
    dialog_recorder: list[dict] = []
    stub = _make_stub(
        view_column_values=[True],
        button_clicks={harness.button_key: True},
    )
    _run_tab(harness, stub, item, dialog_recorder=dialog_recorder.append)

    assert dialog_recorder == [item], (
        f"{harness.label}: the dialog must be invoked exactly once with the checked row's item; "
        f"got {len(dialog_recorder)} call(s)"
    )


# ---------------------------------------------------------------------------
# Test 5 — CSV export DOES NOT include the View column (Questions tab)
# ---------------------------------------------------------------------------


def test_csv_export_does_not_include_view_column():
    """The on-screen ``edited_df`` carries the ``View`` column (for the
    checkbox), but the CSV export DataFrame must NOT — exports build
    from a separate aggregator (``get_question_audit_export``). Each
    table's export path is unchanged from PRs #135 / #167."""
    from views import admin_analytics

    # We render the Questions tab AND click ``audit_export_btn`` to
    # cause the export DataFrame to be built and pushed through
    # ``st.download_button``. The stub captures the CSV bytes so we can
    # decode them and assert on columns.
    captured_export_dfs: list[pd.DataFrame] = []

    class _Stub:
        def __init__(self):
            self.session_state = {"user_role": 0}
            self.secrets = {"agent_logging": {"mode": "full"}}
            self.components = MagicMock()
            self.components.v1 = MagicMock()
            self.components.v1.html = MagicMock()
            self.column_config = _ColumnConfig()

        def multiselect(self, *_a, key=None, **_kw):
            if key is not None:
                self.session_state.setdefault(key, [])
            return []

        def text_input(self, *_a, key=None, **_kw):
            if key is not None:
                self.session_state.setdefault(key, "")
            return ""

        def selectbox(self, *_a, options=None, index=0, key=None, **_kw):
            opts = list(options or [])
            val = opts[index] if opts else None
            if key:
                self.session_state.setdefault(key, val)
            return val

        def number_input(self, *_a, key=None, **_kw):
            if key is not None:
                self.session_state.setdefault(key, 1)
            return 1

        def columns(self, spec, **_kw):
            n = spec if isinstance(spec, int) else len(spec)
            return [MagicMock() for _ in range(n)]

        def container(self, **_kw):
            return MagicMock(__enter__=lambda s: s, __exit__=lambda *a: None)

        def divider(self):
            pass

        def subheader(self, *_a, **_kw):
            pass

        def info(self, *_a, **_kw):
            pass

        def caption(self, *_a, **_kw):
            pass

        def markdown(self, *_a, **_kw):
            pass

        def write(self, *_a, **_kw):
            pass

        def code(self, *_a, **_kw):
            pass

        def warning(self, *_a, **_kw):
            pass

        def error(self, *_a, **_kw):
            pass

        def data_editor(self, df, **_kw):
            # No rows checked — irrelevant for export.
            return df

        def dataframe(self, _df, **_kw):
            return MagicMock()

        def button(self, _label="", key=None, **_kw):
            return key == "audit_export_btn"

        def download_button(self, _label, data=None, **_kw):
            from io import StringIO

            try:
                df = pd.read_csv(StringIO(data.decode("utf-8")))
                captured_export_dfs.append(df)
            except Exception:
                pass

        def dialog(self, _title):
            def _decorator(fn):
                return fn

            return _decorator

    stub = _Stub()
    page_payload = {"items": [_make_question_item(user_message_id=1)], "total": 1}
    export_rows = [_make_question_item(user_message_id=1)]

    with (
        patch.object(admin_analytics, "st", stub),
        patch.object(
            admin_analytics,
            "_cached_audit_filter_options",
            return_value={"usernames": [], "orgs": []},
        ),
        patch("orm.logging_functions.get_question_audit_page", return_value=page_payload),
        patch("orm.logging_functions.get_question_audit_export", return_value=export_rows),
        patch("orm.functions.get_all_users", return_value=[]),
    ):
        admin_analytics._render_audit_trail_tab(30)

    assert captured_export_dfs, "download_button data must have decoded as CSV"
    df = captured_export_dfs[-1]
    assert "View" not in df.columns, f"CSV export must NOT include the View column; saw columns={list(df.columns)}"
