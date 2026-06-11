"""Tests for the Questions audit tab View checkbox dialog (#155 / #170).

Covers:
  (a) Dialog auto-opens when exactly one row's ``View`` checkbox is
      ticked (Epic #169 / #170 swapped the dataframe selection trigger
      for a ``st.data_editor`` + labeled checkbox column that auto-opens
      the dialog on tick).
  (b) Deep-link round-trip — setting `manage_users_pref_user_id` causes
      Manage Users to pre-populate `mu_selected_label` with the target user.
  (c) `role_can_see_query_details` returning False hides the SQL block and
      the full DataFrame inside the dialog body while keeping question /
      summary / error visible.
  (d) `[agent_logging].mode = "disabled"` makes the dialog unreachable
      (``st.data_editor`` is not invoked; only the read-only ``st.dataframe``
      branch runs).
  (e) The dialog renders all rows from `dataframe_preview`, not just the
      first 5 (regression on the old expander `.head(5)` truncation).
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from orm.models import RoleTypeEnum


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_item(
    *,
    user_message_id: int = 42,
    user_id: int = 7,
    username: str = "alice",
    organization: str | None = "Acme",
    question: str = "How many patients today?",
    sql_text: str | None = "SELECT count(*) FROM patient",
    summary_text: str | None = "There are 12 patients.",
    dataframe_preview: str | None = None,
    error_text: str | None = None,
    elapsed_seconds: float = 1.25,
    asked_at: datetime | None = None,
) -> dict:
    """Build a single audit item matching the shape of
    orm.logging_functions._enrich_with_assistant_aggregates output."""
    if dataframe_preview is None:
        # 7 rows so a `.head(5)` truncation would lose 2 rows.
        df = pd.DataFrame({"x": list(range(7)), "y": [chr(ord("a") + i) for i in range(7)]})
        dataframe_preview = df.to_json()
    return {
        "asked_at": asked_at or datetime(2026, 6, 1, 12, 0, 0),
        "user_id": user_id,
        "username": username,
        "organization": organization,
        "question": question,
        "sql_text": sql_text,
        "status": "Error" if error_text else "Success",
        "elapsed_seconds": elapsed_seconds,
        "summary_text": summary_text,
        "dataframe_preview": dataframe_preview,
        "error_text": error_text,
        "user_message_id": user_message_id,
    }


class _RecordingStreamlit:
    """A minimal recording stand-in for the `st` module that captures the
    sequence of code/dataframe/write/markdown calls made by the dialog
    body so we can assert what was rendered.

    Only the surface used by `_render_audit_question_dialog` is implemented.
    """

    def __init__(self, user_role: int | None = RoleTypeEnum.ADMIN.value):
        self.session_state: dict = {"user_role": user_role}
        self.code_calls: list[tuple[str, str]] = []  # (content, language)
        self.dataframe_calls: list[pd.DataFrame] = []
        self.write_calls: list = []
        self.markdown_calls: list[str] = []
        self.caption_calls: list[str] = []
        self.button_clicked_key: str | None = None
        self.switch_page_target: str | None = None

        # components.v1.html stub
        self.components = MagicMock()
        self.components.v1 = MagicMock()
        self.components.v1.html = MagicMock()

    # ---- Streamlit surface --------------------------------------------------
    def markdown(self, body, **_kwargs):
        self.markdown_calls.append(str(body))

    def write(self, body, **_kwargs):
        self.write_calls.append(body)

    def code(self, body, language: str = "text", **_kwargs):
        self.code_calls.append((str(body), language))

    def dataframe(self, df, **_kwargs):
        # We only record real DataFrames passed by the dialog body.
        if isinstance(df, pd.DataFrame):
            self.dataframe_calls.append(df)
        return MagicMock()

    def caption(self, body, **_kwargs):
        self.caption_calls.append(str(body))

    def divider(self):  # no-op
        pass

    def button(self, label, key=None, **_kwargs):
        # Default: button NOT clicked. Tests can override via
        # `force_button_click_key` to simulate a click.
        clicked = key is not None and key == getattr(self, "_force_click_key", None)
        if clicked:
            self.button_clicked_key = key
        return clicked

    def switch_page(self, target):
        self.switch_page_target = target

    # Decorators must be no-ops: we want to call the wrapped function directly.
    def dialog(self, _title):
        def _decorator(fn):
            return fn

        return _decorator


# ---------------------------------------------------------------------------
# (c) Role gate — full DataFrame and SQL hidden when role cannot see details
# ---------------------------------------------------------------------------


class TestDialogRoleGate:
    def test_admin_sees_sql_and_dataframe(self):
        """When role_can_see_query_details returns True, the dialog renders the
        SQL block AND the full DataFrame."""
        from views import admin_analytics

        rec = _RecordingStreamlit(user_role=RoleTypeEnum.ADMIN.value)
        item = _make_item()

        with patch.object(admin_analytics, "st", rec):
            with patch(
                "agent.observability_gate.role_can_see_query_details",
                return_value=True,
            ):
                admin_analytics._render_audit_question_dialog_body(item)

        # SQL block rendered
        sql_codes = [body for body, lang in rec.code_calls if lang == "sql"]
        assert sql_codes, "Expected SQL block to be rendered for admin role"
        assert "SELECT count(*)" in sql_codes[0]

        # Full DataFrame rendered (all 7 rows from the preview JSON)
        assert rec.dataframe_calls, "Expected DataFrame to be rendered for admin role"
        assert len(rec.dataframe_calls[0]) == 7, "Dialog must render the FULL stored DataFrame, not .head(5)"

    def test_non_admin_hides_sql_and_dataframe_keeps_other_sections(self):
        """When role_can_see_query_details returns False, hide SQL and full
        DataFrame; question / summary / error remain visible."""
        from views import admin_analytics

        rec = _RecordingStreamlit(user_role=RoleTypeEnum.DOCTOR.value)
        item = _make_item(error_text="Something went wrong on the warehouse.")

        with patch.object(admin_analytics, "st", rec):
            with patch(
                "agent.observability_gate.role_can_see_query_details",
                return_value=False,
            ):
                admin_analytics._render_audit_question_dialog_body(item)

        # No SQL code block
        sql_codes = [body for body, lang in rec.code_calls if lang == "sql"]
        assert sql_codes == [], "SQL must be hidden when role cannot see query details"

        # No DataFrame rendered
        assert rec.dataframe_calls == [], "Full DataFrame must be hidden when role cannot see query details"

        # Question + summary still visible
        write_str = " ".join(str(x) for x in rec.write_calls)
        assert "How many patients today?" in write_str
        assert "There are 12 patients." in write_str

        # Error block still rendered as plain text
        text_codes = [body for body, lang in rec.code_calls if lang == "text"]
        assert any("Something went wrong" in body for body in text_codes)


# ---------------------------------------------------------------------------
# Full DataFrame rendering (Testable Outcome: "renders all rows")
# ---------------------------------------------------------------------------


class TestDialogFullDataFrame:
    def test_dialog_renders_all_rows_not_just_first_five(self):
        """For a DataFrame with > 5 rows, the dialog must render ALL rows.
        Regression on the old expander's `.head(5)` truncation."""
        from views import admin_analytics

        df = pd.DataFrame({"n": list(range(12))})
        item = _make_item(dataframe_preview=df.to_json())
        rec = _RecordingStreamlit(user_role=RoleTypeEnum.ADMIN.value)

        with patch.object(admin_analytics, "st", rec):
            with patch(
                "agent.observability_gate.role_can_see_query_details",
                return_value=True,
            ):
                admin_analytics._render_audit_question_dialog_body(item)

        assert rec.dataframe_calls, "Expected a DataFrame to be rendered"
        rendered = rec.dataframe_calls[0]
        assert len(rendered) == 12, f"Dialog must render all 12 rows, got {len(rendered)}"


# ---------------------------------------------------------------------------
# Outbound deep-link (Manage Users)
# ---------------------------------------------------------------------------


class TestDialogManageUsersDeepLink:
    def test_button_click_sets_pref_and_switches_page(self):
        """Clicking 'View user in Manage Users →' must set
        `manage_users_pref_user_id` to the row's user_id and call
        st.switch_page('views/admin.py')."""
        from views import admin_analytics

        item = _make_item(user_id=42, user_message_id=99)
        rec = _RecordingStreamlit(user_role=RoleTypeEnum.ADMIN.value)
        rec._force_click_key = f"audit_dialog_goto_users_{item['user_message_id']}"

        with patch.object(admin_analytics, "st", rec):
            with patch(
                "agent.observability_gate.role_can_see_query_details",
                return_value=True,
            ):
                admin_analytics._render_audit_question_dialog_body(item)

        assert rec.session_state.get("manage_users_pref_user_id") == 42
        assert rec.switch_page_target == "views/admin.py"
        # JS tab shim must be emitted so the Admin page lands on the Users tab.
        assert rec.components.v1.html.called, "JS tab-selection shim must be emitted on deep-link click"

    def test_button_not_clicked_does_not_set_pref(self):
        """If the user does not click the button, no session_state mutation
        and no page switch."""
        from views import admin_analytics

        item = _make_item(user_id=42, user_message_id=99)
        rec = _RecordingStreamlit(user_role=RoleTypeEnum.ADMIN.value)
        # No _force_click_key set → button.return_value False

        with patch.object(admin_analytics, "st", rec):
            with patch(
                "agent.observability_gate.role_can_see_query_details",
                return_value=True,
            ):
                admin_analytics._render_audit_question_dialog_body(item)

        assert "manage_users_pref_user_id" not in rec.session_state
        assert rec.switch_page_target is None


# ---------------------------------------------------------------------------
# Inbound deep-link consumption (Manage Users side)
# ---------------------------------------------------------------------------


class TestManageUsersInboundPrefill:
    def test_pref_user_id_pre_populates_mu_selected_label(self):
        """Setting `manage_users_pref_user_id` in session_state must cause
        `render()` to look up the matching user and pre-populate
        `mu_selected_label` with the formatted option label."""
        from views import admin_users

        rec_session: dict = {"manage_users_pref_user_id": 7}

        users_list = [
            {
                "id": 7,
                "username": "alice",
                "first_name": "Alice",
                "last_name": "Smith",
                "role_name": "Admin",
                "email": "a@a.io",
                "organization": "Acme",
                "theme": "healthelink",
            },
            {
                "id": 8,
                "username": "bob",
                "first_name": "Bob",
                "last_name": "Jones",
                "role_name": "Doctor",
                "email": "b@b.io",
                "organization": "Beta",
                "theme": "healthelink",
            },
        ]

        # We don't need to render the full UI — we only need to verify the
        # prefill block at the top of render() does the right thing before
        # anything else runs. Patch out the heavy downstream Streamlit and
        # DB surface so we can exit early.
        with (
            patch.object(admin_users, "st") as mock_st,
            patch.object(admin_users, "get_all_users", return_value=users_list),
            patch.object(admin_users, "get_all_user_roles", return_value=[]),
            patch.object(admin_users, "get_user_stats_for_all_users", return_value={}),
        ):
            # session_state should behave like a dict for the popped key + the
            # "mu_selected_label not in session_state" check.
            mock_st.session_state = rec_session
            # Make the rest of render() abort early by raising on st.columns
            mock_st.columns.side_effect = RuntimeError("abort after prefill")

            with pytest.raises(RuntimeError, match="abort after prefill"):
                admin_users.render()

        assert rec_session.get("mu_selected_label") == "alice (Alice Smith) - Admin", (
            "mu_selected_label must match the option label format from admin_users.py"
        )
        # Pref must have been consumed (popped).
        assert "manage_users_pref_user_id" not in rec_session

    def test_pref_does_not_clobber_existing_selection(self):
        """If `mu_selected_label` is already set, the inbound pref must NOT
        overwrite it (user's current selection wins)."""
        from views import admin_users

        rec_session: dict = {
            "manage_users_pref_user_id": 7,
            "mu_selected_label": "bob (Bob Jones) - Doctor",
        }

        users_list = [
            {
                "id": 7,
                "username": "alice",
                "first_name": "Alice",
                "last_name": "Smith",
                "role_name": "Admin",
                "email": None,
                "organization": None,
                "theme": "healthelink",
            }
        ]

        with (
            patch.object(admin_users, "st") as mock_st,
            patch.object(admin_users, "get_all_users", return_value=users_list),
            patch.object(admin_users, "get_all_user_roles", return_value=[]),
            patch.object(admin_users, "get_user_stats_for_all_users", return_value={}),
        ):
            mock_st.session_state = rec_session
            mock_st.columns.side_effect = RuntimeError("abort after prefill")

            with pytest.raises(RuntimeError, match="abort after prefill"):
                admin_users.render()

        assert rec_session.get("mu_selected_label") == "bob (Bob Jones) - Doctor", (
            "Existing mu_selected_label must not be overwritten by the inbound pref"
        )
        # Even when it doesn't apply, the pref key is still popped (consume-on-read).
        assert "manage_users_pref_user_id" not in rec_session


# ---------------------------------------------------------------------------
# (d) agent_logging.mode = "disabled" — dialog unreachable
# ---------------------------------------------------------------------------


class TestAgentLoggingDisabled:
    def test_disabled_mode_uses_dataframe_not_data_editor(self):
        """When `[agent_logging].mode = "disabled"`, the audit tab must
        render via the read-only ``st.dataframe`` branch and NOT call
        ``st.data_editor`` — so there's no checkbox column, making the
        dialog unreachable (defense in depth — disabled mode should
        produce no rows in the first place). The read-only display
        must also drop the ``View`` column entirely (it would be a
        confusing always-False checkbox)."""
        from views import admin_analytics

        dataframe_call_count = [0]
        data_editor_call_count = [0]
        captured_readonly_dfs = []

        class _Stub:
            def __init__(self):
                self.session_state = {}
                self.secrets = {"agent_logging": {"mode": "disabled"}}
                self.components = MagicMock()
                self.components.v1 = MagicMock()
                self.components.v1.html = MagicMock()
                self.column_config = MagicMock()

            def multiselect(self, *_a, key=None, **_kw):
                self.session_state.setdefault(key, [])
                return []

            def text_input(self, *_a, key=None, **_kw):
                self.session_state.setdefault(key, "")
                return ""

            def selectbox(self, *_a, options=None, index=0, key=None, **_kw):
                opts = list(options or [])
                val = opts[index] if opts else None
                if key:
                    self.session_state.setdefault(key, val)
                return val

            def number_input(self, *_a, key=None, **_kw):
                self.session_state.setdefault(key, 1)
                return 1

            def columns(self, spec):
                n = spec if isinstance(spec, int) else len(spec)
                return [MagicMock() for _ in range(n)]

            def dataframe(self, df, **_kwargs):
                dataframe_call_count[0] += 1
                # Capture only DataFrames whose key indicates the audit grid
                # so we don't grab unrelated read-only tables.
                if _kwargs.get("key") == "audit_dataframe":
                    captured_readonly_dfs.append(df)
                return MagicMock()

            def data_editor(self, df, **_kwargs):
                data_editor_call_count[0] += 1
                return df

            def info(self, *_a, **_kw):
                pass

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

            def divider(self):
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

        stub = _Stub()
        filter_options = {"usernames": ["alice"], "orgs": ["Acme"]}
        page_payload = {
            "items": [_make_item()],
            "total": 1,
        }

        with (
            patch.object(admin_analytics, "st", stub),
            patch.object(admin_analytics, "_cached_audit_filter_options", return_value=filter_options),
            patch("orm.logging_functions.get_question_audit_page", return_value=page_payload),
            patch("orm.functions.get_all_users", return_value=[]),
        ):
            admin_analytics._render_audit_trail_tab(30)

        assert data_editor_call_count[0] == 0, (
            "When agent_logging.mode == 'disabled', st.data_editor must NOT be called — "
            "the dialog must be unreachable. The audit table should fall through to the "
            "read-only st.dataframe branch."
        )
        assert dataframe_call_count[0] >= 1, (
            "In disabled mode, the table should still render via the read-only st.dataframe branch."
        )
        # The read-only DataFrame must NOT carry a meaningless ``View`` column.
        assert captured_readonly_dfs, "Expected the audit grid to render via st.dataframe in disabled mode"
        assert "View" not in captured_readonly_dfs[0].columns, (
            "In disabled mode the read-only display must drop the View column entirely; "
            f"saw columns={list(captured_readonly_dfs[0].columns)}"
        )

    def test_full_mode_uses_data_editor_with_view_checkbox_column(self):
        """When `[agent_logging].mode` is anything other than 'disabled'
        (default 'full'), the audit tab must call ``st.data_editor`` and
        configure a leading ``View`` ``CheckboxColumn`` so the
        auto-open-on-tick workflow is wired up."""
        from views import admin_analytics

        captured_data_editor_kwargs: list[dict] = []

        class _Stub:
            def __init__(self):
                self.session_state = {}
                self.secrets = {"agent_logging": {"mode": "full"}}
                self.components = MagicMock()
                self.components.v1 = MagicMock()
                self.components.v1.html = MagicMock()
                # column_config attribute: returning the call args lets
                # downstream code (the tab) inspect what was configured.
                self.column_config = MagicMock()

            def multiselect(self, *_a, key=None, **_kw):
                self.session_state.setdefault(key, [])
                return []

            def text_input(self, *_a, key=None, **_kw):
                self.session_state.setdefault(key, "")
                return ""

            def selectbox(self, *_a, options=None, index=0, key=None, **_kw):
                opts = list(options or [])
                val = opts[index] if opts else None
                if key:
                    self.session_state.setdefault(key, val)
                return val

            def number_input(self, *_a, key=None, **_kw):
                self.session_state.setdefault(key, 1)
                return 1

            def columns(self, spec):
                n = spec if isinstance(spec, int) else len(spec)
                return [MagicMock() for _ in range(n)]

            def data_editor(self, df, **kwargs):
                captured_data_editor_kwargs.append(kwargs)
                # Return the DataFrame unchanged so the tab can read
                # ``edited_df["View"]`` (all False by default).
                return df

            def dataframe(self, _df, **_kw):
                return MagicMock()

            def info(self, *_a, **_kw):
                pass

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

            def divider(self):
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

        stub = _Stub()
        filter_options = {"usernames": ["alice"], "orgs": ["Acme"]}
        page_payload = {"items": [_make_item()], "total": 1}

        with (
            patch.object(admin_analytics, "st", stub),
            patch.object(admin_analytics, "_cached_audit_filter_options", return_value=filter_options),
            patch("orm.logging_functions.get_question_audit_page", return_value=page_payload),
            patch("orm.functions.get_all_users", return_value=[]),
        ):
            admin_analytics._render_audit_trail_tab(30)

        assert captured_data_editor_kwargs, "Expected st.data_editor to be called"
        kw = captured_data_editor_kwargs[0]
        assert kw.get("key") == "audit_dataframe"
        cc = kw.get("column_config")
        assert cc is not None and "View" in cc, (
            f"Expected a 'View' entry in column_config; saw keys={list((cc or {}).keys())}"
        )


# ---------------------------------------------------------------------------
# (a) Auto-open dialog on tick; 0 ticks → no dialog; sticky tick → no re-fire
# ---------------------------------------------------------------------------


def _make_questions_stub(*, view_column_values: list[bool], initial_session: dict | None = None):
    """Build a Streamlit stub for ``_render_audit_trail_tab`` that simulates
    the new (Epic #169 / #170) data_editor + auto-open-on-tick flow.

    ``view_column_values`` controls which rows the user has 'checked' on
    the ``View`` column. The stub seeds those values back into ``edited_df``
    so the production code's ``edited_df[edited_df['View']]`` filter picks
    them up just like Streamlit would after a checkbox toggle.

    ``initial_session`` seeds ``session_state`` — used by the "sticky
    tick across reruns" tests to pre-populate
    ``audit_dialog_open_user_message_id``.
    """
    import pandas as pd

    class _Stub:
        def __init__(self):
            self.session_state = dict(initial_session or {})
            self.secrets = {"agent_logging": {"mode": "full"}}
            self.components = MagicMock()
            self.components.v1 = MagicMock()
            self.components.v1.html = MagicMock()
            self.column_config = MagicMock()
            self.captured_data_editor_kwargs: list[dict] = []

        def multiselect(self, *_a, key=None, **_kw):
            self.session_state.setdefault(key, [])
            return []

        def text_input(self, *_a, key=None, **_kw):
            self.session_state.setdefault(key, "")
            return ""

        def selectbox(self, *_a, options=None, index=0, key=None, **_kw):
            opts = list(options or [])
            val = opts[index] if opts else None
            if key:
                self.session_state.setdefault(key, val)
            return val

        def number_input(self, *_a, key=None, **_kw):
            self.session_state.setdefault(key, 1)
            return 1

        def columns(self, spec):
            n = spec if isinstance(spec, int) else len(spec)
            return [MagicMock() for _ in range(n)]

        def data_editor(self, df, **kwargs):
            self.captured_data_editor_kwargs.append(kwargs)
            # Simulate the user toggling checkboxes: rebuild ``View`` with
            # the per-test values so the production filter picks them up.
            out = df.copy()
            if "View" in out.columns and view_column_values:
                # Pad / truncate to the row count.
                vals = list(view_column_values) + [False] * max(0, len(out) - len(view_column_values))
                out["View"] = vals[: len(out)]
            return out

        def dataframe(self, _df, **_kw):
            return MagicMock()

        def info(self, *_a, **_kw):
            pass

        def button(self, *_a, key=None, **_kw):
            # No button participates in the auto-open flow — Prev/Next/
            # Export buttons still render but do nothing here.
            return False

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

        def download_button(self, *_a, **_kw):
            pass

        def dialog(self, _title):
            def _decorator(fn):
                return fn

            return _decorator

    return _Stub(), pd  # pd returned for the test's local convenience


class TestSelectionStateTrigger:
    def test_one_tick_auto_opens_dialog(self):
        """When exactly one row's View checkbox is ticked, the audit tab
        must set ``audit_dialog_open_user_message_id`` and auto-invoke
        the dialog function for that row — no button click needed."""
        from views import admin_analytics

        item = _make_item(user_message_id=314)
        dialog_calls: list[dict] = []

        stub, _pd = _make_questions_stub(view_column_values=[True])

        def _fake_dialog(passed_item):
            dialog_calls.append(passed_item)

        with (
            patch.object(admin_analytics, "st", stub),
            patch.object(
                admin_analytics,
                "_cached_audit_filter_options",
                return_value={"usernames": [], "orgs": []},
            ),
            patch.object(admin_analytics, "_render_audit_question_dialog", side_effect=_fake_dialog),
            patch(
                "orm.logging_functions.get_question_audit_page",
                return_value={"items": [item], "total": 1},
            ),
            patch("orm.functions.get_all_users", return_value=[]),
        ):
            admin_analytics._render_audit_trail_tab(30)

        assert dialog_calls == [item], "Dialog must be invoked for the ticked row's item"
        assert stub.session_state.get("audit_dialog_open_user_message_id") == 314

    def test_zero_ticks_does_not_open_dialog(self):
        """If no row is ticked, the dialog must NOT open and the per-tab
        ``open_id`` gate is cleared so the next tick of the same row will
        reopen the dialog cleanly."""
        from views import admin_analytics

        item = _make_item(user_message_id=314)

        # No row is ticked. Pre-populate the gate so we can assert it gets
        # cleared (the "untick clears gate" half of the contract).
        stub, _pd = _make_questions_stub(
            view_column_values=[False],
            initial_session={"audit_dialog_open_user_message_id": 314},
        )

        with (
            patch.object(admin_analytics, "st", stub),
            patch.object(
                admin_analytics,
                "_cached_audit_filter_options",
                return_value={"usernames": [], "orgs": []},
            ),
            patch.object(admin_analytics, "_render_audit_question_dialog") as dialog_mock,
            patch(
                "orm.logging_functions.get_question_audit_page",
                return_value={"items": [item], "total": 1},
            ),
            patch("orm.functions.get_all_users", return_value=[]),
        ):
            admin_analytics._render_audit_trail_tab(30)

        dialog_mock.assert_not_called()
        assert "audit_dialog_open_user_message_id" not in stub.session_state, (
            "Zero ticks must clear the open_id gate so the same row can be re-ticked"
        )

    def test_sticky_tick_across_rerun_does_not_refire(self):
        """If the checkbox stays ticked between reruns (Streamlit
        ``data_editor`` retains the True value) the dialog must NOT
        re-fire — the per-tab gate stops it. Without the gate, every
        rerun would trigger another dialog."""
        from views import admin_analytics

        item = _make_item(user_message_id=314)

        # Row is still ticked, AND the gate already records this row as
        # open from a prior rerun.
        stub, _pd = _make_questions_stub(
            view_column_values=[True],
            initial_session={"audit_dialog_open_user_message_id": 314},
        )

        with (
            patch.object(admin_analytics, "st", stub),
            patch.object(
                admin_analytics,
                "_cached_audit_filter_options",
                return_value={"usernames": [], "orgs": []},
            ),
            patch.object(admin_analytics, "_render_audit_question_dialog") as dialog_mock,
            patch(
                "orm.logging_functions.get_question_audit_page",
                return_value={"items": [item], "total": 1},
            ),
            patch("orm.functions.get_all_users", return_value=[]),
        ):
            admin_analytics._render_audit_trail_tab(30)

        dialog_mock.assert_not_called()
        # Gate stays set to the same id (sticky tick).
        assert stub.session_state.get("audit_dialog_open_user_message_id") == 314
