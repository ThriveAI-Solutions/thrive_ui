"""Tests for the Scope filter UI on the Questions audit tab.

Epic #166 / Feature #167. Specifically:

  - The Scope multiselect renders with the 4 expected options and a stable
    ``audit_scope_filter`` session key (matching the existing
    ``audit_user_filter`` / ``audit_org_filter`` convention).
  - The selected scopes are plumbed into the ``filters`` dict and reach
    ``get_question_audit_page`` and ``get_question_audit_export``.
  - The Scope chip participates in the existing filter-signature reset
    logic — changing it resets ``audit_page_num`` to 1.
  - The grid table_rows include a ``"Scope"`` column.
  - The CSV export DataFrame includes a ``"Scope"`` column.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd

from orm.models import RoleTypeEnum


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


def _make_item(*, user_message_id: int = 42, scope: str = "Patient") -> dict:
    """One item matching ``_enrich_with_assistant_aggregates``' shape, now
    carrying a ``scope`` label."""
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
        "scope": scope,
    }


def _make_stub(
    *,
    secrets_mode: str = "full",
    multiselect_returns: dict | None = None,
    button_returns: dict | None = None,
    initial_session_state: dict | None = None,
):
    """Build a recording streamlit stub for ``_render_audit_trail_tab``.

    Parameters
    ----------
    multiselect_returns : dict keyed by ``key=`` value. Lets tests inject
        the "user picked these" values without faking widget interaction.
    button_returns : dict keyed by ``key=`` value. Lets tests fire the CSV
        export button (``audit_export_btn``) or the View Selected button
        (``audit_questions_view_selected_btn``) on demand.

    Epic #169 / #170: the Questions audit grid is now ``st.data_editor``,
    so ``table_rows`` are captured from ``data_editor(df, ...)`` rather
    than ``dataframe(df, ...)``. The read-only ``st.dataframe`` branch
    (agent_logging.mode = 'disabled') is also captured for completeness.
    """

    captured_multiselect_kwargs: list[dict] = []
    captured_table_rows: list[list[dict]] = []
    captured_download_kwargs: list[dict] = []
    captured_export_df: list[pd.DataFrame] = []
    captured_data_editor_kwargs: list[dict] = []

    multiselect_returns = multiselect_returns or {}
    button_returns = button_returns or {}

    class _Stub:
        def __init__(self):
            self.session_state = dict(initial_session_state or {})
            self.session_state.setdefault("user_role", RoleTypeEnum.ADMIN.value)
            self.secrets = {"agent_logging": {"mode": secrets_mode}}
            self.components = MagicMock()
            self.components.v1 = MagicMock()
            self.components.v1.html = MagicMock()
            self.column_config = MagicMock()

        # -- widgets -------------------------------------------------------
        def multiselect(self, label, options=None, key=None, **kw):
            captured_multiselect_kwargs.append(
                {
                    "label": label,
                    "options": list(options or []),
                    "key": key,
                    "kwargs": kw,
                }
            )
            val = multiselect_returns.get(key, [])
            if key is not None:
                self.session_state.setdefault(key, val)
                self.session_state[key] = val
            return val

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

        def columns(self, spec):
            n = spec if isinstance(spec, int) else len(spec)
            return [MagicMock() for _ in range(n)]

        def data_editor(self, df, **kw):
            captured_data_editor_kwargs.append(kw)
            try:
                captured_table_rows.append(df.to_dict(orient="records"))
            except Exception:
                captured_table_rows.append([])
            # Return the DataFrame unchanged (no rows checked by default,
            # so the View Selected button will be disabled and no dialog
            # branch fires).
            return df

        def dataframe(self, df, **_kw):
            # Disabled-mode read-only branch.
            try:
                captured_table_rows.append(df.to_dict(orient="records"))
            except Exception:
                captured_table_rows.append([])
            return MagicMock()

        def info(self, *_a, **_kw):
            pass

        def button(self, *_a, key=None, **_kw):
            return button_returns.get(key, False)

        def caption(self, *_a, **_kw):
            pass

        def markdown(self, *_a, **_kw):
            pass

        def write(self, *_a, **_kw):
            pass

        def code(self, *_a, **_kw):
            pass

        def divider(self):
            pass

        def warning(self, *_a, **_kw):
            pass

        def error(self, *_a, **_kw):
            pass

        def download_button(self, _label, data=None, **kw):
            captured_download_kwargs.append({"data": data, **kw})
            # Reconstruct the export DataFrame from the encoded bytes so
            # tests can assert on column shape without depending on the
            # exact CSV serialization.
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
    return stub, {
        "multiselect": captured_multiselect_kwargs,
        "table_rows": captured_table_rows,
        "download_kwargs": captured_download_kwargs,
        "export_df": captured_export_df,
        "data_editor_kwargs": captured_data_editor_kwargs,
    }


# ---------------------------------------------------------------------------
# (1) Scope multiselect is rendered with the right options + key
# ---------------------------------------------------------------------------


def test_scope_multiselect_options_and_key():
    """The Scope chip must render with options
    ``[Patient, Pop Health, Other, Legacy/Unknown]`` keyed by
    ``audit_scope_filter`` — the spec's stable session key."""
    from views import admin_analytics

    stub, captures = _make_stub()
    page_payload = {"items": [_make_item()], "total": 1}

    with (
        patch.object(admin_analytics, "st", stub),
        patch.object(admin_analytics, "_cached_audit_filter_options", return_value={"usernames": [], "orgs": []}),
        patch("orm.logging_functions.get_question_audit_page", return_value=page_payload),
        patch("orm.functions.get_all_users", return_value=[]),
    ):
        admin_analytics._render_audit_trail_tab(30)

    scope_chip = next(
        (m for m in captures["multiselect"] if m["key"] == "audit_scope_filter"),
        None,
    )
    assert scope_chip is not None, "audit_scope_filter multiselect must be rendered"
    assert scope_chip["options"] == [
        "Patient",
        "Pop Health",
        "Other",
        "Legacy/Unknown",
    ]
    assert scope_chip["label"] == "Scope"


# ---------------------------------------------------------------------------
# (2) Selected scopes are plumbed into the filters dict for both page + CSV
# ---------------------------------------------------------------------------


def test_selected_scope_is_passed_into_get_question_audit_page():
    from views import admin_analytics

    stub, _ = _make_stub(multiselect_returns={"audit_scope_filter": ["Patient", "Pop Health"]})
    page_payload = {"items": [], "total": 0}

    captured_filters: list[dict] = []

    def _fake_page(filters, page=1, page_size=50):
        captured_filters.append(filters)
        return page_payload

    with (
        patch.object(admin_analytics, "st", stub),
        patch.object(admin_analytics, "_cached_audit_filter_options", return_value={"usernames": [], "orgs": []}),
        patch("orm.logging_functions.get_question_audit_page", side_effect=_fake_page),
        patch("orm.functions.get_all_users", return_value=[]),
    ):
        admin_analytics._render_audit_trail_tab(30)

    assert captured_filters, "get_question_audit_page must have been called"
    last = captured_filters[-1]
    assert last["scopes"] == ["Patient", "Pop Health"]


def test_csv_export_passes_scope_filter_through():
    from views import admin_analytics

    stub, captures = _make_stub(
        multiselect_returns={"audit_scope_filter": ["Pop Health"]},
        button_returns={"audit_export_btn": True},
    )
    page_payload = {"items": [_make_item(scope="Pop Health")], "total": 1}
    export_rows = [_make_item(user_message_id=42, scope="Pop Health")]

    captured_export_filters: list[dict] = []

    def _fake_export(filters):
        captured_export_filters.append(filters)
        return export_rows

    with (
        patch.object(admin_analytics, "st", stub),
        patch.object(admin_analytics, "_cached_audit_filter_options", return_value={"usernames": [], "orgs": []}),
        patch("orm.logging_functions.get_question_audit_page", return_value=page_payload),
        patch("orm.logging_functions.get_question_audit_export", side_effect=_fake_export),
        patch("orm.functions.get_all_users", return_value=[]),
    ):
        admin_analytics._render_audit_trail_tab(30)

    assert captured_export_filters, "Export filter dict must have been built"
    assert captured_export_filters[-1]["scopes"] == ["Pop Health"]

    # And the resulting CSV DataFrame must carry the Scope column.
    assert captures["export_df"], "download_button data must have decoded as CSV"
    df = captures["export_df"][-1]
    assert "Scope" in df.columns
    assert df["Scope"].tolist() == ["Pop Health"]


# ---------------------------------------------------------------------------
# (3) Filter signature includes Scope — changing it resets audit_page_num
# ---------------------------------------------------------------------------


def test_changing_scope_resets_audit_page_num_to_1():
    """The filter-signature reset at views/admin_analytics.py:419-422 must
    notice when ``audit_scope_filter`` changes."""
    from views import admin_analytics

    # First render: user has Patient selected; page bumped to 3 by hand
    # (mimicking the user clicking Next twice).
    stub1, _ = _make_stub(
        multiselect_returns={"audit_scope_filter": ["Patient"]},
        initial_session_state={"audit_page_num": 3},
    )
    page_payload = {"items": [], "total": 0}

    with (
        patch.object(admin_analytics, "st", stub1),
        patch.object(admin_analytics, "_cached_audit_filter_options", return_value={"usernames": [], "orgs": []}),
        patch("orm.logging_functions.get_question_audit_page", return_value=page_payload),
        patch("orm.functions.get_all_users", return_value=[]),
    ):
        admin_analytics._render_audit_trail_tab(30)

    signature_with_patient = stub1.session_state["audit_filter_signature"]
    # Page reset on filter change relative to the initial empty signature.
    assert stub1.session_state["audit_page_num"] == 1

    # Second render: same starting page (3), but Scope flipped to Pop
    # Health. Carry the prior filter signature forward to simulate a
    # genuine "same session, just changed the chip" rerun.
    stub2, _ = _make_stub(
        multiselect_returns={"audit_scope_filter": ["Pop Health"]},
        initial_session_state={
            "audit_page_num": 3,
            "audit_filter_signature": signature_with_patient,
        },
    )
    with (
        patch.object(admin_analytics, "st", stub2),
        patch.object(admin_analytics, "_cached_audit_filter_options", return_value={"usernames": [], "orgs": []}),
        patch("orm.logging_functions.get_question_audit_page", return_value=page_payload),
        patch("orm.functions.get_all_users", return_value=[]),
    ):
        admin_analytics._render_audit_trail_tab(30)

    assert stub2.session_state["audit_page_num"] == 1
    assert stub2.session_state["audit_filter_signature"] != signature_with_patient


def test_stable_scope_does_not_reset_page():
    """If the user only paginated (nothing about the filter changed), the
    page bump should survive. This catches a stale-signature bug where the
    Scope chip's session-state quirks accidentally triggered a reset."""
    from views import admin_analytics

    # Compute the expected signature once via a no-op render so the second
    # render shares it.
    bootstrap_stub, _ = _make_stub(
        multiselect_returns={"audit_scope_filter": ["Patient"]},
    )
    page_payload = {"items": [], "total": 0}

    with (
        patch.object(admin_analytics, "st", bootstrap_stub),
        patch.object(admin_analytics, "_cached_audit_filter_options", return_value={"usernames": [], "orgs": []}),
        patch("orm.logging_functions.get_question_audit_page", return_value=page_payload),
        patch("orm.functions.get_all_users", return_value=[]),
    ):
        admin_analytics._render_audit_trail_tab(30)
    stable_signature = bootstrap_stub.session_state["audit_filter_signature"]

    # Now the "user paginated" rerun. Same Scope, but page_num is 4 and
    # the signature is already cached — reset must NOT fire.
    stub, _ = _make_stub(
        multiselect_returns={"audit_scope_filter": ["Patient"]},
        initial_session_state={
            "audit_page_num": 4,
            "audit_filter_signature": stable_signature,
        },
    )
    with (
        patch.object(admin_analytics, "st", stub),
        patch.object(admin_analytics, "_cached_audit_filter_options", return_value={"usernames": [], "orgs": []}),
        patch("orm.logging_functions.get_question_audit_page", return_value=page_payload),
        patch("orm.functions.get_all_users", return_value=[]),
    ):
        admin_analytics._render_audit_trail_tab(30)
    assert stub.session_state["audit_page_num"] == 4


# ---------------------------------------------------------------------------
# (4) Grid surface — Scope column appears in the rendered table_rows
# ---------------------------------------------------------------------------


def test_grid_table_rows_include_scope_column_between_status_and_elapsed():
    from views import admin_analytics

    stub, captures = _make_stub()
    page_payload = {
        "items": [
            _make_item(user_message_id=1, scope="Patient"),
            _make_item(user_message_id=2, scope="Pop Health"),
        ],
        "total": 2,
    }

    with (
        patch.object(admin_analytics, "st", stub),
        patch.object(admin_analytics, "_cached_audit_filter_options", return_value={"usernames": [], "orgs": []}),
        patch("orm.logging_functions.get_question_audit_page", return_value=page_payload),
        patch("orm.functions.get_all_users", return_value=[]),
    ):
        admin_analytics._render_audit_trail_tab(30)

    assert captures["table_rows"], "dataframe must have been rendered"
    rows = captures["table_rows"][-1]
    assert all("Scope" in r for r in rows)
    assert [r["Scope"] for r in rows] == ["Patient", "Pop Health"]

    # And Scope must land between Status and Elapsed (s) (spec).
    cols = list(rows[0].keys())
    status_idx = cols.index("Status")
    scope_idx = cols.index("Scope")
    elapsed_idx = cols.index("Elapsed (s)")
    assert status_idx < scope_idx < elapsed_idx, f"Scope must sit between Status and Elapsed; got column order {cols}"


# ---------------------------------------------------------------------------
# (5) Empty Scope selection means no filter is plumbed
# ---------------------------------------------------------------------------


def test_empty_scope_selection_passes_none_to_backend():
    """Default state (no chips picked) must NOT send ``scopes`` as an empty
    list to the backend — that would be ambiguous. Send ``None`` so the
    backend's existing ``filters.get("scopes") or None`` short-circuits."""
    from views import admin_analytics

    stub, _ = _make_stub()  # default: multiselect returns []
    captured: list[dict] = []

    def _fake_page(filters, page=1, page_size=50):
        captured.append(filters)
        return {"items": [], "total": 0}

    with (
        patch.object(admin_analytics, "st", stub),
        patch.object(admin_analytics, "_cached_audit_filter_options", return_value={"usernames": [], "orgs": []}),
        patch("orm.logging_functions.get_question_audit_page", side_effect=_fake_page),
        patch("orm.functions.get_all_users", return_value=[]),
    ):
        admin_analytics._render_audit_trail_tab(30)

    assert captured[-1]["scopes"] is None
