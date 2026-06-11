"""Tests for Epic #169 / Feature #170 — the labeled ``View`` checkbox
column shared by all three audit tables (Questions, Admin Actions, User
Activity).

Per the post-PR-#187 refactor (Feature #170), the trigger is **auto-open
on tick**: ticking exactly one ``View`` checkbox opens the corresponding
detail dialog immediately — there is no ``View Selected`` button.

Covers, parametrised across the 3 tabs where possible:

  - The ``View`` column is configured as ``st.column_config.CheckboxColumn``
    with the spec's label and help tooltip.
  - All non-View data columns are configured with ``disabled=True``.
  - Ticking exactly one row auto-fires the dialog with the matching row
    payload; 0 and >1 ticks do NOT fire the dialog.
  - With the per-tab ``open_id`` already set in ``session_state``,
    re-rendering with the same row still ticked does NOT re-fire the
    dialog (gate prevents re-fire on every rerun).
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
    data_editor key, dialog gate session key + id field, dialog function
    name) so the parametrised tests can drive any of the 3 audit tabs
    from a single body."""

    def __init__(
        self,
        *,
        label: str,
        render_fn_name: str,
        item_factory,
        data_editor_key: str,
        prev_view_checks_key: str,
        dialog_id_key: str,
        id_field: str,
        dialog_attr: str,
        loader_patches: list,
    ):
        self.label = label
        self.render_fn_name = render_fn_name
        self.item_factory = item_factory
        self.data_editor_key = data_editor_key
        self.prev_view_checks_key = prev_view_checks_key
        self.dialog_id_key = dialog_id_key
        self.id_field = id_field
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
            prev_view_checks_key="audit_questions_prev_view_checks",
            dialog_id_key="audit_dialog_open_user_message_id",
            id_field="user_message_id",
            dialog_attr="_render_audit_question_dialog",
            loader_patches=_questions_loader_patches,
        ),
        _TabHarness(
            label="admin_actions",
            render_fn_name="_render_audit_tab",
            item_factory=lambda i: _make_admin_action_item(id=i),
            data_editor_key="audit_actions_dataframe",
            prev_view_checks_key="audit_actions_prev_view_checks",
            dialog_id_key="audit_actions_dialog_open_id",
            id_field="id",
            dialog_attr="_render_admin_action_dialog",
            loader_patches=_actions_loader_patches,
        ),
        _TabHarness(
            label="user_activity",
            render_fn_name="_render_activity_tab",
            item_factory=lambda i: _make_user_activity_item(id=i),
            data_editor_key="audit_activity_dataframe",
            prev_view_checks_key="audit_activity_prev_view_checks",
            dialog_id_key="audit_activity_dialog_open_id",
            id_field="id",
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
    initial_session: dict | None = None,
):
    """Build a streamlit stub for any of the three tab render functions.

    ``view_column_values`` is the list of per-row checkbox values written
    back into ``edited_df`` (simulating user toggles).
    ``initial_session`` seeds the stub's ``session_state`` dict — used
    by the "checkbox stays ticked across rerun" tests to pre-populate
    the tab's ``open_id`` gate.
    """

    class _Stub:
        def __init__(self):
            self.session_state = dict(initial_session or {})
            self.secrets = {"agent_logging": {"mode": "full"}}
            self.components = MagicMock()
            self.components.v1 = MagicMock()
            self.components.v1.html = MagicMock()
            self.column_config = _ColumnConfig()
            self.captured_data_editor_kwargs: list[dict] = []
            self.rerun = MagicMock()

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
            # No buttons participate in the auto-open flow anymore. The
            # Prev/Next/Export buttons still render — we just return False
            # so they do nothing.
            return False

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
# Test 3 — Auto-open: exactly one tick fires the dialog
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("harness", _all_harnesses(), ids=lambda h: h.label)
def test_one_tick_auto_opens_dialog(harness):
    """With exactly one ``View`` checkbox ticked the corresponding
    dialog function is auto-invoked with the matching row's payload —
    no button click is needed."""
    item = harness.item_factory(314)
    dialog_recorder: list[dict] = []
    stub = _make_stub(view_column_values=[True])
    _run_tab(harness, stub, item, dialog_recorder=dialog_recorder.append)

    assert dialog_recorder == [item], (
        f"{harness.label}: ticking exactly one row must auto-fire the dialog; got {len(dialog_recorder)} call(s)"
    )
    # The per-tab open_id gate must be set to the row's id so subsequent
    # reruns don't re-fire the dialog while the checkbox stays ticked.
    assert stub.session_state.get(harness.dialog_id_key) == item[harness.id_field]


# ---------------------------------------------------------------------------
# Test 4 — Zero ticks: no dialog, gate cleared
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("harness", _all_harnesses(), ids=lambda h: h.label)
def test_zero_ticks_does_not_open_dialog(harness):
    """No rows ticked → the dialog must not fire. The per-tab gate is
    also cleared so a subsequent tick of the same row reopens the
    dialog cleanly."""
    item = harness.item_factory(7)
    dialog_recorder: list[dict] = []
    stub = _make_stub(
        view_column_values=[False],
        # Pre-populate the gate to verify it gets cleared on zero ticks.
        initial_session={harness.dialog_id_key: item[harness.id_field]},
    )
    _run_tab(harness, stub, item, dialog_recorder=dialog_recorder.append)

    assert dialog_recorder == [], (
        f"{harness.label}: zero ticks must NOT open the dialog; got {len(dialog_recorder)} call(s)"
    )
    assert harness.dialog_id_key not in stub.session_state, (
        f"{harness.label}: zero ticks must clear the {harness.dialog_id_key!r} gate so the "
        "same row can be re-ticked to reopen the dialog"
    )


# ---------------------------------------------------------------------------
# Test 5 — Multi-tick: no dialog fires
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("harness", _all_harnesses(), ids=lambda h: h.label)
@pytest.mark.parametrize("n_checked", [2, 3], ids=["two", "three"])
def test_multi_tick_does_not_open_dialog(harness, n_checked):
    """More than one row ticked → no dialog fires. User must untick the
    extras down to exactly one for the dialog to open."""
    items = [harness.item_factory(i) for i in range(1, n_checked + 1)]
    view_vals = [True] * n_checked
    dialog_recorder: list[dict] = []

    stub = _make_stub(view_column_values=view_vals)

    def _swap_loaders(_items):
        return harness.loader_patches(items)

    swapped = _TabHarness(
        label=harness.label,
        render_fn_name=harness.render_fn_name,
        item_factory=harness.item_factory,
        data_editor_key=harness.data_editor_key,
        prev_view_checks_key=harness.prev_view_checks_key,
        dialog_id_key=harness.dialog_id_key,
        id_field=harness.id_field,
        dialog_attr=harness.dialog_attr,
        loader_patches=_swap_loaders,
    )

    _run_tab(swapped, stub, items[0], dialog_recorder=dialog_recorder.append)

    assert dialog_recorder == [], (
        f"{harness.label}: {n_checked} ticks must NOT open the dialog; got {len(dialog_recorder)} call(s)"
    )


# ---------------------------------------------------------------------------
# Test 6 — Checkbox stays ticked across rerun: dialog does NOT re-fire
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("harness", _all_harnesses(), ids=lambda h: h.label)
def test_checkbox_stays_ticked_across_rerun_does_not_refire(harness):
    """If the checkbox stays ticked between reruns (Streamlit
    ``data_editor`` retains the row's True value) the dialog must NOT
    re-fire — the per-tab ``open_id`` gate stops it. This is the
    critical regression: without the gate, every rerun would trigger
    another dialog."""
    item = harness.item_factory(42)
    dialog_recorder: list[dict] = []

    # Simulate "the dialog was already opened in a prior rerun" by
    # pre-populating the per-tab gate. The checkbox is still ticked.
    stub = _make_stub(
        view_column_values=[True],
        initial_session={harness.dialog_id_key: item[harness.id_field]},
    )
    _run_tab(harness, stub, item, dialog_recorder=dialog_recorder.append)

    assert dialog_recorder == [], (
        f"{harness.label}: with the per-tab gate already set to this row's id and the "
        "checkbox still ticked, the dialog must NOT re-fire on this rerun. "
        f"Got {len(dialog_recorder)} unwanted call(s)."
    )
    # Gate stays set to the same id.
    assert stub.session_state.get(harness.dialog_id_key) == item[harness.id_field]


# ---------------------------------------------------------------------------
# Test 7 — Untick then re-tick reopens the dialog
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("harness", _all_harnesses(), ids=lambda h: h.label)
def test_untick_then_retick_reopens_dialog(harness):
    """Rerun A: row ticked → dialog opens, gate set. Rerun B: row
    unticked → gate cleared. Rerun C: row re-ticked → dialog opens
    again. This is what "Allow re-ticking the same row to reopen" in
    the spec means."""
    item = harness.item_factory(99)

    # Rerun A — row ticked, fresh session.
    a_calls: list = []
    stub_a = _make_stub(view_column_values=[True])
    _run_tab(harness, stub_a, item, dialog_recorder=a_calls.append)
    assert a_calls == [item], f"{harness.label}: rerun A must open the dialog"
    assert stub_a.session_state.get(harness.dialog_id_key) == item[harness.id_field]

    # Rerun B — row unticked. Carry session_state forward. The cross-tab
    # claim flag is reset by ``admin_audit.render`` in production at the
    # top of every rerun, before the inner ``st.tabs`` evaluate; we
    # simulate that here by dropping it on each new rerun.
    b_session = dict(stub_a.session_state)
    b_session.pop("_audit_dialog_claimed_this_rerun", None)
    b_calls: list = []
    stub_b = _make_stub(
        view_column_values=[False],
        initial_session=b_session,
    )
    _run_tab(harness, stub_b, item, dialog_recorder=b_calls.append)
    assert b_calls == [], f"{harness.label}: rerun B (unticked) must not fire"
    assert harness.dialog_id_key not in stub_b.session_state, f"{harness.label}: untick must clear the gate"

    # Rerun C — row re-ticked.
    c_session = dict(stub_b.session_state)
    c_session.pop("_audit_dialog_claimed_this_rerun", None)
    c_calls: list = []
    stub_c = _make_stub(
        view_column_values=[True],
        initial_session=c_session,
    )
    _run_tab(harness, stub_c, item, dialog_recorder=c_calls.append)
    assert c_calls == [item], f"{harness.label}: rerun C (re-ticked after untick) must reopen the dialog"


# ---------------------------------------------------------------------------
# Test 8 — CSV export DOES NOT include the View column (Questions tab)
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


# ---------------------------------------------------------------------------
# Test 9 — Single-select enforcement (multi-tick auto-unticks older rows)
# ---------------------------------------------------------------------------


def _build_multi_item_harness(harness, items):
    """Clone ``harness`` so its loader returns ``items`` (multi-row case).
    Mirrors the swap pattern from ``test_multi_tick_does_not_open_dialog``.
    """

    def _swap_loaders(_items):
        return harness.loader_patches(items)

    return _TabHarness(
        label=harness.label,
        render_fn_name=harness.render_fn_name,
        item_factory=harness.item_factory,
        data_editor_key=harness.data_editor_key,
        prev_view_checks_key=harness.prev_view_checks_key,
        dialog_id_key=harness.dialog_id_key,
        id_field=harness.id_field,
        dialog_attr=harness.dialog_attr,
        loader_patches=_swap_loaders,
    )


@pytest.mark.parametrize("harness", _all_harnesses(), ids=lambda h: h.label)
def test_two_ticks_unticks_older_row_and_reruns(harness):
    """User had row 1 ticked. They tick row 2 as well. Single-select
    enforcement must:
    (a) clear row 1's ``View`` entry from the data_editor's widget-state
        ``edited_rows`` (so the next render shows row 1 as unticked),
    (b) call ``st.rerun()`` exactly once,
    (c) NOT open any dialog on this rerun — the dialog opens on the NEXT
        rerun when ``checked_count == 1`` is true again.
    """
    items = [harness.item_factory(i) for i in range(1, 4)]  # 3 rows
    swapped = _build_multi_item_harness(harness, items)

    initial_session = {
        harness.prev_view_checks_key: [1],
        harness.data_editor_key: {
            "edited_rows": {1: {"View": True}, 2: {"View": True}},
        },
    }

    calls: list = []
    stub = _make_stub(
        view_column_values=[False, True, True],
        initial_session=initial_session,
    )
    _run_tab(swapped, stub, items[0], dialog_recorder=calls.append)

    edited_rows = stub.session_state[harness.data_editor_key].get("edited_rows", {})
    # Row 1 (older tick) must be cleared.
    assert "View" not in edited_rows.get(1, {}), (
        f"{harness.label}: row 1 (older tick) must be removed from edited_rows; saw edited_rows={edited_rows!r}"
    )
    # Row 2 (newer tick) must be preserved.
    assert edited_rows.get(2, {}).get("View") is True, f"{harness.label}: row 2 (newer tick) must remain in edited_rows"

    stub.rerun.assert_called_once()
    assert calls == [], (
        f"{harness.label}: multi-tick rerun must not open dialog "
        f"(dialog opens on the NEXT rerun once edits are reflected)"
    )

    # ``prev_view_checks`` is updated to the kept row.
    assert stub.session_state.get(harness.prev_view_checks_key) == [2], (
        f"{harness.label}: prev_view_checks must reflect the kept row"
    )


@pytest.mark.parametrize("harness", _all_harnesses(), ids=lambda h: h.label)
def test_three_ticks_keeps_a_newly_checked_row(harness):
    """User had row 0 ticked previously. They add ticks on rows 1 and 2
    in the same interaction (now 3 total). Enforcement keeps one newly-
    checked row and unticks the older row 0."""
    items = [harness.item_factory(i) for i in range(1, 4)]  # 3 rows
    swapped = _build_multi_item_harness(harness, items)

    initial_session = {
        harness.prev_view_checks_key: [0],
        harness.data_editor_key: {
            "edited_rows": {
                0: {"View": True},
                1: {"View": True},
                2: {"View": True},
            },
        },
    }

    calls: list = []
    stub = _make_stub(
        view_column_values=[True, True, True],
        initial_session=initial_session,
    )
    _run_tab(swapped, stub, items[0], dialog_recorder=calls.append)

    edited_rows = stub.session_state[harness.data_editor_key].get("edited_rows", {})
    # Row 0 (only prev-checked row) must be cleared.
    assert "View" not in edited_rows.get(0, {}), (
        f"{harness.label}: row 0 (only prev-checked) must be cleared; saw edited_rows={edited_rows!r}"
    )
    # At least one of {1, 2} must remain (the keeper).
    remaining = [idx for idx in (1, 2) if edited_rows.get(idx, {}).get("View") is True]
    assert remaining, f"{harness.label}: at least one newly-checked row must remain; saw edited_rows={edited_rows!r}"

    stub.rerun.assert_called_once()
    assert calls == []


@pytest.mark.parametrize("harness", _all_harnesses(), ids=lambda h: h.label)
def test_single_tick_does_not_trigger_rerun(harness):
    """Sanity check: the new enforcement block is a no-op when only one
    row is ticked. ``st.rerun()`` must NOT be called and the dialog must
    open normally."""
    item = harness.item_factory(99)

    calls: list = []
    stub = _make_stub(view_column_values=[True])
    _run_tab(harness, stub, item, dialog_recorder=calls.append)

    stub.rerun.assert_not_called()
    assert calls == [item]
    assert stub.session_state.get(harness.prev_view_checks_key) == [0]


@pytest.mark.parametrize("harness", _all_harnesses(), ids=lambda h: h.label)
def test_zero_ticks_resets_prev_view_checks_to_empty(harness):
    """When the user unticks everything, ``prev_view_checks`` resets to
    the empty list so a subsequent single tick is treated as fresh."""
    item = harness.item_factory(99)

    initial_session = {harness.prev_view_checks_key: [1]}
    calls: list = []
    stub = _make_stub(
        view_column_values=[False, False, False],
        initial_session=initial_session,
    )
    _run_tab(harness, stub, item, dialog_recorder=calls.append)

    stub.rerun.assert_not_called()
    assert calls == []
    assert stub.session_state.get(harness.prev_view_checks_key) == []
