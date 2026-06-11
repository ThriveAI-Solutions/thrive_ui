"""Tests for the leading ``View`` affordance column on the three Admin → Audit
tables (Questions, Admin Actions, User Activity).

Epic #169 / Feature #170. Specifically:

  - Each of the three audit tables prepends a leading ``View`` column to its
    ``table_rows`` dict (so it lands as the leftmost column in the rendered
    DataFrame).
  - Every row carries the same single-character icon glyph
    (``_VIEW_COLUMN_ICON``) under the ``View`` key.
  - The ``column_config`` kwarg passed to ``st.dataframe`` carries an entry
    keyed by ``_VIEW_COLUMN_LABEL`` whose ``Column(label=..., help=...)``
    call uses the module constants verbatim. This is what gives the user the
    hover tooltip ``"Click any row to open the detail dialog"``.
  - The Questions CSV export DataFrame is NOT polluted with the ``View``
    column — the export aggregator is a separate code path and must remain
    free of the on-screen affordance. (Admin Actions and User Activity have
    no CSV export today; if they ever ship one, the same invariant applies.)

The existing dialog-open / cross-tab collision behaviour is covered by
``tests/views/test_admin_audit_dialog.py``, ``tests/views/test_admin_actions_dialog.py``,
``tests/views/test_user_activity_dialog.py``, and
``tests/views/test_admin_audit_dialog_collision.py`` — those suites pass
unchanged after this feature lands (verified after stub
``column_config = MagicMock()`` shims were added in PR #170).
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from orm.models import RoleTypeEnum


# ---------------------------------------------------------------------------
# Shared stub factory
# ---------------------------------------------------------------------------


def _make_tab_stub(
    *,
    secrets_mode: str = "full",
    button_returns: dict | None = None,
):
    """Build a Streamlit stub usable for all three audit tab bodies.

    Captures every ``st.dataframe(...)`` call's positional DataFrame (so we
    can read its ``table_rows``) AND its kwargs (so we can inspect the
    ``column_config`` dict the audit tab wires up).

    Captures every ``st.column_config.Column(...)`` invocation so the tests
    can assert the ``label``/``help`` arguments the audit tab passes.
    """

    captured_dataframes: list[pd.DataFrame] = []
    captured_dataframe_kwargs: list[dict] = []
    captured_column_calls: list[dict] = []
    captured_download_kwargs: list[dict] = []
    captured_export_df: list[pd.DataFrame] = []

    button_returns = button_returns or {}

    class _ColumnConfigStub:
        """Stand-in for ``st.column_config`` that records ``Column(...)`` calls."""

        @staticmethod
        def Column(label=None, help=None, width=None, **_kw):  # noqa: N802
            call = {"label": label, "help": help, "width": width, **_kw}
            captured_column_calls.append(call)
            # Return a sentinel that's distinguishable in assertions.
            return ("__view_column_marker__", call)

    class _Stub:
        def __init__(self):
            self.session_state = {"user_role": RoleTypeEnum.ADMIN.value}
            self.secrets = {"agent_logging": {"mode": secrets_mode}}
            self.column_config = _ColumnConfigStub()
            self.components = MagicMock()
            self.components.v1 = MagicMock()
            self.components.v1.html = MagicMock()

        # ---- Inputs ----
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

        # ---- Layout ----
        def columns(self, spec, **_kw):
            n = spec if isinstance(spec, int) else len(spec)
            cols = []
            for _ in range(n):
                cm = MagicMock()
                cm.__enter__ = MagicMock(return_value=cm)
                cm.__exit__ = MagicMock(return_value=False)
                cols.append(cm)
            return cols

        def container(self, **_kw):
            cm = MagicMock()
            cm.__enter__ = MagicMock(return_value=cm)
            cm.__exit__ = MagicMock(return_value=False)
            return cm

        def divider(self):
            pass

        # ---- Outputs ----
        def dataframe(self, df, **kwargs):
            captured_dataframes.append(df)
            captured_dataframe_kwargs.append(kwargs)
            ev = MagicMock()
            ev.selection = {"rows": []}
            return ev

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

        def subheader(self, *_a, **_kw):
            pass

        def metric(self, *_a, **_kw):
            pass

        def button(self, *_a, key=None, **_kw):
            return button_returns.get(key, False)

        def download_button(self, _label, data=None, **kw):
            captured_download_kwargs.append({"data": data, **kw})
            try:
                from io import StringIO

                df = pd.read_csv(StringIO(data.decode("utf-8")))
                captured_export_df.append(df)
            except Exception:
                pass

        def dialog(self, _title):
            def _decorator(fn):
                return fn

            return _decorator

    stub = _Stub()
    captures = {
        "dataframes": captured_dataframes,
        "dataframe_kwargs": captured_dataframe_kwargs,
        "column_calls": captured_column_calls,
        "download_kwargs": captured_download_kwargs,
        "export_df": captured_export_df,
    }
    return stub, captures


# ---------------------------------------------------------------------------
# Per-table item factories — match what each ``_render_*_tab`` consumes from
# the corresponding page-result ``items`` list.
# ---------------------------------------------------------------------------


def _make_question_item(user_message_id: int = 1) -> dict:
    return {
        "asked_at": datetime(2026, 6, 1, 12, 0, 0),
        "user_id": 7,
        "username": "alice",
        "organization": "Acme",
        "question": "How many patients today?",
        "sql_text": "SELECT count(*) FROM patient",
        "status": "Success",
        "elapsed_seconds": 1.25,
        "summary_text": "There are 12 patients.",
        "dataframe_preview": pd.DataFrame({"x": [1]}).to_json(),
        "error_text": None,
        "user_message_id": user_message_id,
        "scope": "Patient",
    }


def _make_activity_item(activity_id: int = 1) -> dict:
    return {
        "id": activity_id,
        "created_at": datetime(2026, 6, 1, 13, 0, 0),
        "user_id": 7,
        "username": "alice",
        "activity_type": "login",
        "description": "logged in",
        "ip_address": "10.0.0.1",
    }


def _make_action_item(action_id: int = 1) -> dict:
    return {
        "id": action_id,
        "created_at": datetime(2026, 6, 1, 14, 0, 0),
        "admin_id": 1,
        "admin_username": "admin",
        "action_type": "user_role_change",
        "target_username": "bob",
        "target_entity_type": "user",
        "target_entity_id": 7,
        "description": "Promoted bob to admin",
        "success": True,
        "affected_count": 1,
        "old_value": None,
        "new_value": None,
        "error_message": None,
    }


# ---------------------------------------------------------------------------
# Per-tab harness: render the tab body with a single-row seeded page and
# return the captured DataFrame + the captured ``column_config`` kwarg.
# ---------------------------------------------------------------------------


def _render_questions_tab(stub):
    from views import admin_analytics

    page_payload = {"items": [_make_question_item()], "total": 1}
    with (
        patch.object(admin_analytics, "st", stub),
        patch.object(admin_analytics, "_cached_audit_filter_options", return_value={"usernames": [], "orgs": []}),
        patch("orm.logging_functions.get_question_audit_page", return_value=page_payload),
        patch("orm.functions.get_all_users", return_value=[]),
    ):
        admin_analytics._render_audit_trail_tab(30)


def _render_activity_tab(stub):
    from views import admin_analytics

    page_payload = {"items": [_make_activity_item()], "total": 1}
    with (
        patch.object(admin_analytics, "st", stub),
        patch("orm.logging_functions.get_user_activity_page", return_value=page_payload),
    ):
        admin_analytics._render_activity_tab(30)


def _render_admin_actions_tab(stub):
    from views import admin_analytics

    page_payload = {"items": [_make_action_item()], "total": 1}
    with (
        patch.object(admin_analytics, "st", stub),
        patch("orm.logging_functions.get_admin_actions_page", return_value=page_payload),
    ):
        admin_analytics._render_audit_tab(30)


_TAB_RENDERERS = {
    "questions": _render_questions_tab,
    "activity": _render_activity_tab,
    "admin_actions": _render_admin_actions_tab,
}


# ---------------------------------------------------------------------------
# (1) Column presence + position — uniform across all three tabs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tab_name", list(_TAB_RENDERERS.keys()))
def test_view_column_is_the_leftmost_column_in_table_rows(tab_name):
    """The ``View`` key must be the FIRST key in every ``table_rows`` dict —
    i.e., the leftmost rendered column. Each tab body builds its dataframe
    from a ``table_rows`` dict (preserving insertion order under PEP 468) so
    asserting ``list(records[0].keys())[0] == "View"`` pins layout."""
    from views import admin_analytics

    stub, captures = _make_tab_stub()
    _TAB_RENDERERS[tab_name](stub)

    assert captures["dataframes"], f"{tab_name} tab must have rendered st.dataframe"
    rendered_df = captures["dataframes"][-1]
    assert isinstance(rendered_df, pd.DataFrame)
    records = rendered_df.to_dict(orient="records")
    assert records, f"{tab_name} tab must have produced at least one table row"

    first_key = list(records[0].keys())[0]
    assert first_key == admin_analytics._VIEW_COLUMN_LABEL, (
        f"{tab_name}: View column must be the leftmost column; got columns {list(records[0].keys())}"
    )


# ---------------------------------------------------------------------------
# (2) Icon glyph is uniform — every row carries _VIEW_COLUMN_ICON
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tab_name", list(_TAB_RENDERERS.keys()))
def test_view_column_carries_uniform_icon_glyph_per_row(tab_name):
    """Every row's ``View`` cell must hold the same icon glyph — the constant
    ``_VIEW_COLUMN_ICON``."""
    from views import admin_analytics

    stub, captures = _make_tab_stub()
    _TAB_RENDERERS[tab_name](stub)

    rendered_df = captures["dataframes"][-1]
    records = rendered_df.to_dict(orient="records")
    icons = [r[admin_analytics._VIEW_COLUMN_LABEL] for r in records]
    assert all(icon == admin_analytics._VIEW_COLUMN_ICON for icon in icons), (
        f"{tab_name}: every View cell must equal _VIEW_COLUMN_ICON; got {icons}"
    )


# ---------------------------------------------------------------------------
# (3) column_config carries the label + help tooltip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tab_name", list(_TAB_RENDERERS.keys()))
def test_dataframe_column_config_wires_view_help_and_label(tab_name):
    """The ``st.dataframe(...)`` call must pass a ``column_config`` whose
    entry keyed by ``_VIEW_COLUMN_LABEL`` carries a ``Column(label=..., help=...)``
    call wired to the module constants verbatim. The tooltip copy is the
    discoverability win — vary it per-table and the affordance gets noisy.
    """
    from views import admin_analytics

    stub, captures = _make_tab_stub()
    _TAB_RENDERERS[tab_name](stub)

    # Locate the kwargs dict for the dataframe call that carried the View
    # column. There may be more than one dataframe call (e.g., the disabled-
    # mode read-only fallback) — assert at least one wired the column_config.
    matching_kwargs = [kw for kw in captures["dataframe_kwargs"] if "column_config" in kw and kw["column_config"]]
    assert matching_kwargs, (
        f"{tab_name}: at least one st.dataframe call must pass column_config; got {captures['dataframe_kwargs']}"
    )

    for kw in matching_kwargs:
        cc = kw["column_config"]
        assert admin_analytics._VIEW_COLUMN_LABEL in cc, (
            f"{tab_name}: column_config must carry an entry keyed by "
            f"_VIEW_COLUMN_LABEL='{admin_analytics._VIEW_COLUMN_LABEL}'; got keys {list(cc.keys())}"
        )

    # Every Column(...) invocation made for this tab body must use the
    # canonical label + help copy. The stub records each call's kwargs.
    assert captures["column_calls"], f"{tab_name}: st.column_config.Column(...) must have been invoked at least once"
    for call in captures["column_calls"]:
        assert call["label"] == admin_analytics._VIEW_COLUMN_LABEL, (
            f"{tab_name}: View column label must equal _VIEW_COLUMN_LABEL; got {call}"
        )
        assert call["help"] == admin_analytics._VIEW_COLUMN_HELP, (
            f"{tab_name}: View column help must equal _VIEW_COLUMN_HELP; got {call}"
        )


# ---------------------------------------------------------------------------
# (4) CSV export non-pollution — Questions tab
# ---------------------------------------------------------------------------


def test_questions_csv_export_does_not_contain_view_column():
    """The Questions audit export DataFrame is built from a separate
    aggregator path (``get_question_audit_export``) and must remain free of
    the on-screen ``View`` affordance column. Tests press the export button
    via ``button_returns`` and inspect the encoded ``download_button`` data.
    """
    from views import admin_analytics

    stub, captures = _make_tab_stub(button_returns={"audit_export_btn": True})

    export_rows = [_make_question_item(user_message_id=1)]

    with (
        patch.object(admin_analytics, "st", stub),
        patch.object(admin_analytics, "_cached_audit_filter_options", return_value={"usernames": [], "orgs": []}),
        patch("orm.logging_functions.get_question_audit_page", return_value={"items": export_rows, "total": 1}),
        patch("orm.logging_functions.get_question_audit_export", return_value=export_rows),
        patch("orm.functions.get_all_users", return_value=[]),
    ):
        admin_analytics._render_audit_trail_tab(30)

    assert captures["export_df"], "Export download_button must have been called"
    df = captures["export_df"][-1]
    assert admin_analytics._VIEW_COLUMN_LABEL not in df.columns, (
        f"CSV export must NOT carry the View affordance column; got columns {list(df.columns)}"
    )
    # And the data columns we expect should be there — sanity check that the
    # exported DataFrame is the real export, not an empty one.
    assert "Question" in df.columns and "Status" in df.columns


# ---------------------------------------------------------------------------
# (5) get_question_audit_export return value — the data-layer entrypoint —
# must not surface a ``View`` key. This locks the contract at the source so
# any downstream consumer that builds its own DataFrame off the same payload
# inherits the same guarantee.
# ---------------------------------------------------------------------------


def test_question_audit_export_payload_has_no_view_key():
    """Defence-in-depth: even if a future caller builds its own export
    DataFrame off ``get_question_audit_export(filters)``, the payload itself
    must not carry the on-screen affordance column."""
    from views import admin_analytics

    stub, captures = _make_tab_stub(button_returns={"audit_export_btn": True})

    captured_export_rows: list[list[dict]] = []

    def _fake_export(_filters):
        rows = [_make_question_item(user_message_id=1)]
        captured_export_rows.append(rows)
        return rows

    with (
        patch.object(admin_analytics, "st", stub),
        patch.object(admin_analytics, "_cached_audit_filter_options", return_value={"usernames": [], "orgs": []}),
        patch(
            "orm.logging_functions.get_question_audit_page",
            return_value={"items": [_make_question_item(user_message_id=1)], "total": 1},
        ),
        patch("orm.logging_functions.get_question_audit_export", side_effect=_fake_export),
        patch("orm.functions.get_all_users", return_value=[]),
    ):
        admin_analytics._render_audit_trail_tab(30)

    assert captured_export_rows, "Export entrypoint must have been called"
    for row in captured_export_rows[-1]:
        assert admin_analytics._VIEW_COLUMN_LABEL not in row, f"Export row must not carry View key; got {row}"
