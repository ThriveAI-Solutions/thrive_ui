"""Tests for the User Activity tab row-click dialog (#157).

Covers:
  (a) Dialog opens when ``audit_activity_dialog_open_id`` is set and closes
      when the dataframe selection is cleared.
  (b) ``user_agent`` renders in the dialog (surfaced for the first time).
  (c) Old/new JSON renders both columns when both present, only the populated
      side when one is null, skipped when both are null.
  (d) Malformed JSON in ``old_value``/``new_value`` falls back to ``st.text``
      without raising.
  (e) Deep-link "View user in Manage Users →" button is visible when
      ``user_id`` is non-null and hidden when null (failed-login rows).
  (f) Deep-link round-trip — setting ``manage_users_pref_user_id`` causes
      Manage Users to render with the target user pre-selected (asserts
      against the consumer added by #155).
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_item(
    *,
    id: int = 42,
    user_id: int | None = 7,
    username: str = "alice",
    activity_type: str = "settings_change",
    description: str = "Changed theme",
    old_value: str | None = None,
    new_value: str | None = None,
    ip_address: str | None = "127.0.0.1",
    user_agent: str | None = "Mozilla/5.0 RecordingTest",
    created_at: datetime | None = None,
) -> dict:
    """Build a single UserActivity item matching ``get_user_activity_page`` output."""
    return {
        "id": id,
        "created_at": created_at or datetime(2026, 6, 1, 12, 0, 0),
        "user_id": user_id,
        "username": username,
        "activity_type": activity_type,
        "description": description,
        "old_value": old_value,
        "new_value": new_value,
        "ip_address": ip_address,
        "user_agent": user_agent,
    }


class _RecordingStreamlit:
    """A minimal recording stand-in for the ``st`` module used by the dialog body.

    Only the surface used by ``_render_user_activity_dialog_body`` is implemented.
    """

    def __init__(self):
        self.session_state: dict = {}
        self.markdown_calls: list[str] = []
        self.write_calls: list = []
        self.caption_calls: list[str] = []
        self.text_calls: list[str] = []
        self.json_calls: list = []  # the python object passed to st.json
        self.columns_call_count = 0
        self.button_clicked_key: str | None = None
        self.button_calls: list[str] = []  # keys of every button rendered
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

    def text(self, body, **_kwargs):
        self.text_calls.append(str(body))

    def caption(self, body, **_kwargs):
        self.caption_calls.append(str(body))

    def json(self, obj, **_kwargs):
        self.json_calls.append(obj)

    def columns(self, spec, **_kwargs):
        self.columns_call_count += 1
        n = spec if isinstance(spec, int) else len(spec)
        # Return context-manager-capable mocks.
        cols = []
        for _ in range(n):
            cm = MagicMock()
            cm.__enter__ = MagicMock(return_value=cm)
            cm.__exit__ = MagicMock(return_value=False)
            cols.append(cm)
        return cols

    def divider(self):  # no-op
        pass

    def button(self, label, key=None, **_kwargs):
        if key is not None:
            self.button_calls.append(key)
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
# (b) user_agent renders in the dialog
# ---------------------------------------------------------------------------


class TestUserAgentRendered:
    def test_user_agent_is_displayed(self):
        """The dialog must surface ``user_agent`` — the retired
        ``get_recent_activity`` dropped it and no UI rendered it before #157."""
        from views import admin_analytics

        item = _make_item(user_agent="Mozilla/5.0 (Macintosh) CustomBrowser/1.2")
        rec = _RecordingStreamlit()

        with patch.object(admin_analytics, "st", rec):
            admin_analytics._render_user_activity_dialog_body(item)

        all_markdown = " ".join(rec.markdown_calls)
        assert "User Agent" in all_markdown
        assert "Mozilla/5.0 (Macintosh) CustomBrowser/1.2" in all_markdown

    def test_ip_address_is_displayed(self):
        from views import admin_analytics

        item = _make_item(ip_address="192.168.1.42")
        rec = _RecordingStreamlit()

        with patch.object(admin_analytics, "st", rec):
            admin_analytics._render_user_activity_dialog_body(item)

        all_markdown = " ".join(rec.markdown_calls)
        assert "IP Address" in all_markdown
        assert "192.168.1.42" in all_markdown

    def test_header_line_includes_username_and_type(self):
        from views import admin_analytics

        item = _make_item(username="carol", activity_type="login")
        rec = _RecordingStreamlit()

        with patch.object(admin_analytics, "st", rec):
            admin_analytics._render_user_activity_dialog_body(item)

        all_markdown = " ".join(rec.markdown_calls)
        assert "carol" in all_markdown
        assert "login" in all_markdown

    def test_activity_id_caption_rendered(self):
        from views import admin_analytics

        item = _make_item(id=4242)
        rec = _RecordingStreamlit()

        with patch.object(admin_analytics, "st", rec):
            admin_analytics._render_user_activity_dialog_body(item)

        assert any("Activity ID 4242" in c for c in rec.caption_calls), (
            f"Expected 'Activity ID 4242' caption; got {rec.caption_calls}"
        )


# ---------------------------------------------------------------------------
# (c) Old/new JSON columns
# ---------------------------------------------------------------------------


class TestOldNewJsonRendering:
    def test_both_present_renders_two_columns(self):
        """When both old_value and new_value are populated JSON strings,
        both st.json columns are rendered."""
        from views import admin_analytics

        item = _make_item(
            old_value='{"theme": "light"}',
            new_value='{"theme": "dark"}',
        )
        rec = _RecordingStreamlit()

        with patch.object(admin_analytics, "st", rec):
            admin_analytics._render_user_activity_dialog_body(item)

        # Two-column layout used at least once for the diff.
        assert rec.columns_call_count >= 1
        # Both parsed objects passed to st.json.
        assert {"theme": "light"} in rec.json_calls
        assert {"theme": "dark"} in rec.json_calls

    def test_only_new_value_present_renders_only_populated_column(self):
        """When old_value is null but new_value is present (e.g. a fresh
        setting creation), only the populated side renders JSON."""
        from views import admin_analytics

        item = _make_item(old_value=None, new_value='{"theme": "dark"}')
        rec = _RecordingStreamlit()

        with patch.object(admin_analytics, "st", rec):
            admin_analytics._render_user_activity_dialog_body(item)

        # Section is still rendered.
        assert rec.columns_call_count >= 1
        # Only the new value got st.json.
        assert rec.json_calls == [{"theme": "dark"}]

    def test_only_old_value_present_renders_only_populated_column(self):
        from views import admin_analytics

        item = _make_item(old_value='{"theme": "light"}', new_value=None)
        rec = _RecordingStreamlit()

        with patch.object(admin_analytics, "st", rec):
            admin_analytics._render_user_activity_dialog_body(item)

        assert rec.columns_call_count >= 1
        assert rec.json_calls == [{"theme": "light"}]

    def test_both_null_skips_section_entirely(self):
        """When neither old_value nor new_value is present (e.g. a login row),
        the entire old/new diff section is skipped — no columns, no st.json."""
        from views import admin_analytics

        item = _make_item(old_value=None, new_value=None)
        rec = _RecordingStreamlit()

        with patch.object(admin_analytics, "st", rec):
            admin_analytics._render_user_activity_dialog_body(item)

        # No two-column block dedicated to the diff. (The dialog may still
        # call st.columns elsewhere in the future, so just assert no
        # st.json calls for the diff section.)
        assert rec.json_calls == []
        # No "Old Value"/"New Value" markdown headers rendered.
        all_markdown = " ".join(rec.markdown_calls)
        assert "Old Value" not in all_markdown
        assert "New Value" not in all_markdown


# ---------------------------------------------------------------------------
# (d) Malformed JSON fallback
# ---------------------------------------------------------------------------


class TestMalformedJsonFallback:
    def test_malformed_old_value_falls_back_to_text(self):
        """If ``old_value`` is not valid JSON, the dialog must fall back to
        ``st.text`` rather than raising."""
        from views import admin_analytics

        item = _make_item(old_value="not-json-at-all{[", new_value=None)
        rec = _RecordingStreamlit()

        with patch.object(admin_analytics, "st", rec):
            # Must not raise.
            admin_analytics._render_user_activity_dialog_body(item)

        # Raw text was rendered as the fallback.
        assert any("not-json-at-all{[" in t for t in rec.text_calls), (
            f"Expected raw old_value to be rendered via st.text; got text_calls={rec.text_calls}"
        )
        # No st.json call (parse failed).
        assert rec.json_calls == []

    def test_malformed_new_value_falls_back_to_text(self):
        from views import admin_analytics

        item = _make_item(old_value=None, new_value="<<broken>>")
        rec = _RecordingStreamlit()

        with patch.object(admin_analytics, "st", rec):
            admin_analytics._render_user_activity_dialog_body(item)

        assert any("<<broken>>" in t for t in rec.text_calls)
        assert rec.json_calls == []


# ---------------------------------------------------------------------------
# (e) Deep-link button gating on user_id nullability
# ---------------------------------------------------------------------------


class TestDeepLinkButtonGating:
    def test_button_visible_when_user_id_present(self):
        from views import admin_analytics

        item = _make_item(id=99, user_id=7)
        rec = _RecordingStreamlit()

        with patch.object(admin_analytics, "st", rec):
            admin_analytics._render_user_activity_dialog_body(item)

        assert "user_activity_dialog_goto_users_99" in rec.button_calls, (
            f"Expected the deep-link button to be rendered when user_id is non-null; "
            f"buttons rendered: {rec.button_calls}"
        )

    def test_button_hidden_when_user_id_null(self):
        """Failed-login rows have null ``user_id``. The deep-link button must
        be hidden so we don't render an unclickable button."""
        from views import admin_analytics

        item = _make_item(id=100, user_id=None, username="ghost", activity_type="failed_login")
        rec = _RecordingStreamlit()

        with patch.object(admin_analytics, "st", rec):
            admin_analytics._render_user_activity_dialog_body(item)

        assert "user_activity_dialog_goto_users_100" not in rec.button_calls, (
            f"Deep-link button must NOT render when user_id is null; buttons rendered: {rec.button_calls}"
        )


# ---------------------------------------------------------------------------
# (f) Deep-link round-trip
# ---------------------------------------------------------------------------


class TestDeepLinkRoundTrip:
    def test_button_click_sets_pref_and_switches_page(self):
        """Clicking the deep-link button sets ``manage_users_pref_user_id`` and
        calls ``st.switch_page('views/admin.py')``, plus emits the JS tab shim."""
        from views import admin_analytics

        item = _make_item(id=99, user_id=42)
        rec = _RecordingStreamlit()
        rec._force_click_key = "user_activity_dialog_goto_users_99"

        with patch.object(admin_analytics, "st", rec):
            admin_analytics._render_user_activity_dialog_body(item)

        assert rec.session_state.get("manage_users_pref_user_id") == 42
        assert rec.switch_page_target == "views/admin.py"
        assert rec.components.v1.html.called, (
            "JS tab-selection shim must be emitted on deep-link click so the Admin page lands on the Users sub-tab"
        )

    def test_button_not_clicked_does_not_mutate_session(self):
        from views import admin_analytics

        item = _make_item(id=99, user_id=42)
        rec = _RecordingStreamlit()
        # No _force_click_key set — button returns False.

        with patch.object(admin_analytics, "st", rec):
            admin_analytics._render_user_activity_dialog_body(item)

        assert "manage_users_pref_user_id" not in rec.session_state
        assert rec.switch_page_target is None

    def test_pref_round_trip_pre_populates_mu_selected_label(self):
        """End-to-end contract check: emitting ``manage_users_pref_user_id``
        from this dialog must cause ``views/admin_users.render()`` to
        pre-populate ``mu_selected_label`` (the consumer added by #155)."""
        from views import admin_users

        rec_session: dict = {"manage_users_pref_user_id": 42}
        users_list = [
            {
                "id": 42,
                "username": "alice",
                "first_name": "Alice",
                "last_name": "Smith",
                "role_name": "Admin",
                "email": "a@a.io",
                "organization": "Acme",
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

        assert rec_session.get("mu_selected_label") == "alice (Alice Smith) - Admin"
        assert "manage_users_pref_user_id" not in rec_session


# ---------------------------------------------------------------------------
# (a) Selection state → dialog open / close in the activity tab
# ---------------------------------------------------------------------------


def _activity_tab_stub_class():
    """Build a Streamlit stub for ``_render_activity_tab`` testing.

    Returns the class so tests can mutate per-test behaviour.

    Epic #169 / Feature #170: the trigger primitive is now
    ``st.data_editor`` + ``View`` ``CheckboxColumn`` + ``View Selected``
    button. ``view_column_values`` controls which rows are 'checked'
    and ``view_button_clicked`` fires the button.
    """

    class _Stub:
        def __init__(self, *, view_column_values=None, view_button_clicked=False, initial_session=None):
            self.session_state = dict(initial_session or {})
            self.captured_data_editor_kwargs: list[dict] = []
            self._view_column_values = list(view_column_values or [])
            self._view_button_clicked = bool(view_button_clicked)
            self.components = MagicMock()
            self.components.v1 = MagicMock()
            self.components.v1.html = MagicMock()
            self.column_config = MagicMock()

        def selectbox(self, *_a, options=None, index=0, key=None, **_kw):
            opts = list(options or [])
            val = opts[index] if opts else None
            if key:
                self.session_state.setdefault(key, val)
            return val

        def number_input(self, *_a, key=None, **_kw):
            self.session_state.setdefault(key, 1)
            return 1

        def container(self, **_kw):
            cm = MagicMock()
            cm.__enter__ = MagicMock(return_value=cm)
            cm.__exit__ = MagicMock(return_value=False)
            return cm

        def metric(self, *_a, **_kw):
            pass

        def columns(self, spec, **_kw):
            n = spec if isinstance(spec, int) else len(spec)
            cols = []
            for _ in range(n):
                cm = MagicMock()
                cm.__enter__ = MagicMock(return_value=cm)
                cm.__exit__ = MagicMock(return_value=False)
                cols.append(cm)
            return cols

        def data_editor(self, df, **kwargs):
            self.captured_data_editor_kwargs.append(kwargs)
            out = df.copy()
            if "View" in out.columns and self._view_column_values:
                vals = list(self._view_column_values) + [False] * max(0, len(out) - len(self._view_column_values))
                out["View"] = vals[: len(out)]
            return out

        def dataframe(self, _df, **_kwargs):
            return MagicMock()

        def info(self, *_a, **_kw):
            pass

        def button(self, *_a, key=None, **_kw):
            return self._view_button_clicked if key == "audit_activity_view_selected_btn" else False

        def caption(self, *_a, **_kw):
            pass

        def markdown(self, *_a, **_kw):
            pass

        def write(self, *_a, **_kw):
            pass

        def divider(self):
            pass

        def warning(self, *_a, **_kw):
            pass

        def error(self, *_a, **_kw):
            pass

        def subheader(self, *_a, **_kw):
            pass

        def plotly_chart(self, *_a, **_kw):
            pass

        def dialog(self, _title):
            def _decorator(fn):
                return fn

            return _decorator

    return _Stub


class TestActivityTabSelectionTrigger:
    def test_data_editor_wired_with_view_checkbox_column(self):
        """The activity grid must use the Epic #169 / #170 trigger
        primitive: ``st.data_editor`` + leading ``View``
        ``CheckboxColumn`` + a stable key (``audit_activity_dataframe``)."""
        from views import admin_analytics

        item = _make_item(id=999)
        Stub = _activity_tab_stub_class()
        stub = Stub(view_column_values=[])

        with (
            patch.object(admin_analytics, "st", stub),
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
                return_value={"items": [item], "total": 1},
            ),
        ):
            admin_analytics._render_activity_tab(30)

        assert stub.captured_data_editor_kwargs, "Expected st.data_editor to be called"
        kw = stub.captured_data_editor_kwargs[0]
        assert kw.get("key") == "audit_activity_dataframe"
        cc = kw.get("column_config")
        assert cc is not None and "View" in cc, (
            f"Expected a 'View' entry in column_config; saw keys={list((cc or {}).keys())}"
        )

    def test_button_click_with_one_row_checked_opens_dialog(self):
        """With exactly one View checkbox ticked AND the View Selected
        button clicked, ``audit_activity_dialog_open_id`` is set and the
        dialog is invoked with the corresponding item."""
        from views import admin_analytics

        item = _make_item(id=777)
        Stub = _activity_tab_stub_class()
        stub = Stub(view_column_values=[True], view_button_clicked=True)
        dialog_calls: list[dict] = []

        def _fake_dialog(passed_item):
            dialog_calls.append(passed_item)

        with (
            patch.object(admin_analytics, "st", stub),
            patch.object(admin_analytics, "_render_user_activity_dialog", side_effect=_fake_dialog),
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
                return_value={"items": [item], "total": 1},
            ),
        ):
            admin_analytics._render_activity_tab(30)

        assert dialog_calls == [item], "Dialog must be invoked with the checked row's item"
        assert stub.session_state.get("audit_activity_dialog_open_id") == 777

    def test_no_button_click_does_not_open_dialog(self):
        """Checking the box without clicking View Selected must NOT open
        the dialog."""
        from views import admin_analytics

        item = _make_item(id=777)
        Stub = _activity_tab_stub_class()
        # Row checked, button not clicked.
        stub = Stub(view_column_values=[True], view_button_clicked=False)

        with (
            patch.object(admin_analytics, "st", stub),
            patch.object(admin_analytics, "_render_user_activity_dialog") as dialog_mock,
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
                return_value={"items": [item], "total": 1},
            ),
        ):
            admin_analytics._render_activity_tab(30)

        dialog_mock.assert_not_called()
        assert "audit_activity_dialog_open_id" not in stub.session_state


# ---------------------------------------------------------------------------
# "Settings Change Details" expander removed
# ---------------------------------------------------------------------------


def test_settings_change_details_expander_removed():
    """The retired "Settings Change Details" expander must not appear in the
    activity tab anymore — the dialog is the sole detail surface (#157)."""
    from views import admin_analytics

    # Read the source of the function and assert the expander label is gone.
    import inspect

    src = inspect.getsource(admin_analytics._render_activity_tab)
    assert "Settings Change Details" not in src, (
        "The 'Settings Change Details' expander must be retired in favour of the row-click dialog (#157)."
    )
