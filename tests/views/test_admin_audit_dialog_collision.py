"""Regression tests for the Admin → Audit umbrella tab dialog collision (#166).

Background
----------
``views/admin_audit.py:render`` builds an inner ``st.tabs``. The inner tab
list is (post-#190): Queries, By Patient, Admin Actions, User Activity.
Each existing tab wires its grid as ``st.data_editor`` + a leading labeled
``View`` ``CheckboxColumn`` that auto-opens the detail dialog when exactly
one row is ticked. ``st.tabs`` still evaluates *every* tab body on every
rerun, and Streamlit still forbids more than one ``st.dialog`` call per
script run — so the cross-tab claim guard remains as defense in depth.

The per-rerun guard, ``_audit_dialog_claimed_this_rerun``:

* ``views/admin_audit.py:render`` resets the flag to ``False`` at the top of
  every rerun, before ``st.tabs`` is created.
* Each tab dialog branch checks-and-sets the flag *inside* the
  auto-open-on-tick branch, so at most one tab opens a dialog per rerun.

These tests pin both behaviours under the new (data_editor + auto-open)
flow. Epic #190 / Phase 4 retired the Questions tab from the umbrella;
tests that previously used the Questions dialog as the "first tab that
fires" now use the Admin Actions dialog (still in the umbrella) as the
auto-open evidence. The behaviour under test — guard reset, external
dialog detection — is unchanged.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Fixture builders (sample rows + Streamlit stub)
# ---------------------------------------------------------------------------


def _make_question_item(*, user_message_id: int = 1) -> dict:
    """A Questions tab item shaped like ``get_question_audit_page`` output."""
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
    }


def _make_admin_action_item(*, id: int = 1) -> dict:
    """An Admin Actions tab item shaped like ``get_admin_actions_page`` output."""
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
    """A User Activity tab item shaped like ``get_user_activity_page`` output."""
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


class _SharedSessionStub:
    """A Streamlit stub that mimics enough of the surface used by
    ``admin_audit.render`` AND the three tab functions in ``admin_analytics``.

    A single instance must be patched into BOTH ``views.admin_audit.st`` and
    ``views.admin_analytics.st`` so the cross-tab guard
    ``_audit_dialog_claimed_this_rerun`` lives on a SHARED ``session_state``
    dict — that's the whole point of the fix.

    Epic #169 / #170 changed the trigger primitive. Each tab now wires its
    grid as ``st.data_editor`` + a labeled ``View`` checkbox column, and
    auto-opens its dialog when exactly one row's checkbox is ticked.

    ``per_key_view_checked`` lets each tab's data_editor seed the ``View``
    column with a per-row list of booleans, keyed by the data_editor's
    ``key=`` kwarg (one of ``audit_dataframe``, ``audit_actions_dataframe``,
    ``audit_activity_dataframe``).
    """

    # The View column key/value used by all three tabs.
    _VIEW_COL = "View"

    def __init__(
        self,
        *,
        per_key_view_checked: dict | None = None,
        initial_session: dict | None = None,
        secrets: dict | None = None,
    ):
        self.session_state: dict = dict(initial_session or {})
        self._per_key_view_checked = dict(per_key_view_checked or {})
        self.secrets = secrets if secrets is not None else {"agent_logging": {"mode": "full"}}
        # The data_editor kwargs captured per call, for diagnostics.
        self.captured_data_editor_kwargs: list[dict] = []
        # ``components.v1.html`` is invoked by the question dialog body; the
        # tab body itself does not hit it but we expose it for safety.
        self.components = MagicMock()
        self.components.v1 = MagicMock()
        self.components.v1.html = MagicMock()
        # Real ``st.column_config`` is fine to expose as a Mock; the
        # production tab calls ``st.column_config.CheckboxColumn(...)`` /
        # ``TextColumn(...)`` purely to populate the ``column_config`` kwarg
        # we don't introspect here.
        self.column_config = MagicMock()

    # ---- Layout primitives -----------------------------------------------
    def tabs(self, _labels):
        # Return one MagicMock per label, each usable as a context manager.
        result = []
        for _ in _labels:
            cm = MagicMock()
            cm.__enter__ = MagicMock(return_value=cm)
            cm.__exit__ = MagicMock(return_value=False)
            result.append(cm)
        return result

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

    # ---- Input widgets ---------------------------------------------------
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

    def button(self, *_a, key=None, **_kw):
        # No button participates in the auto-open flow — Prev/Next/
        # Export buttons still render but do nothing here.
        return False

    def download_button(self, *_a, **_kw):
        return False

    # ---- Output primitives ----------------------------------------------
    def data_editor(self, df, **kwargs):
        self.captured_data_editor_kwargs.append(kwargs)
        key = kwargs.get("key")
        checks = self._per_key_view_checked.get(key, [])
        out = df.copy()
        if self._VIEW_COL in out.columns and checks:
            vals = list(checks) + [False] * max(0, len(out) - len(checks))
            out[self._VIEW_COL] = vals[: len(out)]
        return out

    def dataframe(self, _df, **_kwargs):
        # Disabled-mode (read-only) branch + the Action Distribution table.
        return MagicMock()

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

    def warning(self, *_a, **_kw):
        pass

    def error(self, *_a, **_kw):
        pass

    def subheader(self, *_a, **_kw):
        pass

    def plotly_chart(self, *_a, **_kw):
        pass

    def metric(self, *_a, **_kw):
        pass

    # ---- Dialog decorator passthrough -----------------------------------
    def dialog(self, _title):
        def _decorator(fn):
            return fn

        return _decorator


def _patches_for_render(
    *,
    stub: _SharedSessionStub,
    questions_items: list,
    actions_items: list,
    activity_items: list,
    spy_q,
    spy_a,
    spy_u,
):
    """Build the full chain of ``patch`` context managers needed to run
    ``admin_audit.render`` end-to-end against the shared stub.

    Returns a list of patchers that the test must enter (e.g. via
    ``contextlib.ExitStack``). Loaders are patched to return controlled rows
    so the three tab functions reach their selection branches deterministically.
    """
    from views import admin_analytics, admin_audit

    patchers = [
        # Share the same stub across both modules so the per-rerun guard key
        # lives on a single session_state dict.
        patch.object(admin_audit, "st", stub),
        patch.object(admin_analytics, "st", stub),
        # Spy on the three dialog functions. They become no-ops here so we
        # don't need their bodies — only call counts matter.
        patch.object(admin_analytics, "_render_audit_question_dialog", side_effect=spy_q),
        patch.object(admin_analytics, "_render_admin_action_dialog", side_effect=spy_a),
        patch.object(admin_analytics, "_render_user_activity_dialog", side_effect=spy_u),
        # Loaders for the Questions tab.
        patch.object(
            admin_analytics,
            "_cached_audit_filter_options",
            return_value={"usernames": [], "orgs": []},
        ),
        patch(
            "orm.logging_functions.get_question_audit_page",
            return_value={"items": list(questions_items), "total": len(questions_items)},
        ),
        patch("orm.functions.get_all_users", return_value=[]),
        # Loaders for the Admin Actions tab.
        patch(
            "orm.logging_functions.get_admin_actions_page",
            return_value={"items": list(actions_items), "total": len(actions_items)},
        ),
        patch(
            "orm.logging_functions.get_admin_action_stats",
            return_value={"total": 1, "user_changes": 0, "training_actions": 0, "failed": 0},
        ),
        patch("orm.logging_functions.get_admin_actions_by_type", return_value=[]),
        # Loaders for the User Activity tab.
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
            return_value={"items": list(activity_items), "total": len(activity_items)},
        ),
    ]
    return patchers


# ---------------------------------------------------------------------------
# (1) At most one dialog per script run when multiple tabs have selections
# ---------------------------------------------------------------------------


class TestCrossTabCollision:
    def test_single_dialog_when_multiple_tabs_have_selections(self):
        """Production repro: Questions tab has row 0 ticked AND Admin
        Actions tab has row 0 ticked AND User Activity tab has row 0
        ticked — all in the same rerun. Without the cross-tab guard,
        every rerun would fire two or three ``st.dialog`` calls and
        Streamlit would raise.

        With the guard, exactly ONE dialog is invoked per rerun.
        """
        from views import admin_audit

        q_item = _make_question_item(user_message_id=314)
        a_item = _make_admin_action_item(id=271)
        u_item = _make_user_activity_item(id=42)

        # All three tabs have row 0 ticked simultaneously — that's the
        # cross-tab collision the guard exists to prevent.
        stub = _SharedSessionStub(
            per_key_view_checked={
                "audit_dataframe": [True],
                "audit_actions_dataframe": [True],
                "audit_activity_dataframe": [True],
            },
        )

        q_calls: list = []
        a_calls: list = []
        u_calls: list = []

        patchers = _patches_for_render(
            stub=stub,
            questions_items=[q_item],
            actions_items=[a_item],
            activity_items=[u_item],
            spy_q=q_calls.append,
            spy_a=a_calls.append,
            spy_u=u_calls.append,
        )

        # Apply all patches then call render. Using a nested ``with`` chain via
        # ExitStack to keep the test readable.
        import contextlib

        with contextlib.ExitStack() as stack:
            for p in patchers:
                stack.enter_context(p)
            admin_audit.render(30)

        total_dialog_calls = len(q_calls) + len(a_calls) + len(u_calls)
        assert total_dialog_calls <= 1, (
            f"At most one dialog may open per script run; saw "
            f"Q={len(q_calls)} A={len(a_calls)} U={len(u_calls)} (={total_dialog_calls} total). "
            f"This is the exact crash mode reported in production."
        )
        # And we should still open exactly one — silently dropping all dialogs
        # would also satisfy ``<= 1`` but would break the feature.
        assert total_dialog_calls == 1, (
            f"Expected exactly one dialog to open (the first tab to claim); got {total_dialog_calls}"
        )

    def test_same_row_reselection_does_not_reopen(self):
        """If the user has row 0 ticked on the Questions tab AND the
        per-tab ``open_id`` gate already records that row, the dialog
        must NOT re-fire on subsequent reruns. This is the gating
        behaviour added in the auto-open refactor — without it, every
        rerun while the checkbox stayed ticked would re-fire the
        dialog."""
        from views import admin_audit

        q_item = _make_question_item(user_message_id=999)

        stub = _SharedSessionStub(
            per_key_view_checked={"audit_dataframe": [True]},
            # Same id already tracked — dialog was opened in a prior rerun.
            initial_session={"audit_dialog_open_user_message_id": 999},
        )

        q_calls: list = []
        a_calls: list = []
        u_calls: list = []

        patchers = _patches_for_render(
            stub=stub,
            questions_items=[q_item],
            actions_items=[],
            activity_items=[],
            spy_q=q_calls.append,
            spy_a=a_calls.append,
            spy_u=u_calls.append,
        )

        import contextlib

        with contextlib.ExitStack() as stack:
            for p in patchers:
                stack.enter_context(p)
            admin_audit.render(30)

        assert q_calls == [], (
            f"Same-row reselection across reruns must not reopen the dialog; saw {len(q_calls)} call(s)"
        )
        # And no other tab should have fired either.
        assert a_calls == []
        assert u_calls == []

    def test_guard_resets_per_rerun(self):
        """The cross-tab claim guard must be cleared at the top of every
        rerun. Two consecutive ``render`` calls in which the user ticks
        the Admin Actions tab's ``View`` checkbox on a different row each
        time must each fire exactly one dialog. (If the guard leaked
        across reruns, the second rerun would silently drop the dialog.)
        """
        from views import admin_audit

        # Two distinct Admin Actions items across the two reruns. Same
        # data_editor key, different ``id``s — so the per-tab
        # ``audit_actions_dialog_open_id`` gate naturally allows the
        # second rerun's dialog to fire (different id → different gate
        # value).
        a_item_1 = _make_admin_action_item(id=111)
        a_item_2 = _make_admin_action_item(id=222)

        # A single shared stub across both reruns so session_state persists,
        # just like real Streamlit. Row 0 is ticked in both reruns; the
        # second rerun's row has a different ``id``.
        stub = _SharedSessionStub(
            per_key_view_checked={"audit_actions_dataframe": [True]},
        )

        # ---- Rerun 1 -----------------------------------------------------
        q1: list = []
        a1: list = []
        u1: list = []

        from views import admin_analytics

        import contextlib

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(admin_audit, "st", stub))
            stack.enter_context(patch.object(admin_analytics, "st", stub))
            stack.enter_context(patch.object(admin_analytics, "_render_audit_question_dialog", side_effect=q1.append))
            stack.enter_context(patch.object(admin_analytics, "_render_admin_action_dialog", side_effect=a1.append))
            stack.enter_context(patch.object(admin_analytics, "_render_user_activity_dialog", side_effect=u1.append))
            stack.enter_context(
                patch.object(
                    admin_analytics,
                    "_cached_audit_filter_options",
                    return_value={"usernames": [], "orgs": []},
                )
            )
            stack.enter_context(
                patch(
                    "orm.logging_functions.get_question_audit_page",
                    return_value={"items": [], "total": 0},
                )
            )
            stack.enter_context(patch("orm.functions.get_all_users", return_value=[]))
            stack.enter_context(
                patch(
                    "orm.logging_functions.get_admin_actions_page",
                    return_value={"items": [a_item_1], "total": 1},
                )
            )
            stack.enter_context(
                patch(
                    "orm.logging_functions.get_admin_action_stats",
                    return_value={"total": 0, "user_changes": 0, "training_actions": 0, "failed": 0},
                )
            )
            stack.enter_context(patch("orm.logging_functions.get_admin_actions_by_type", return_value=[]))
            stack.enter_context(
                patch(
                    "orm.logging_functions.get_activity_stats",
                    return_value={
                        "logins_today": 0,
                        "failed_logins": 0,
                        "settings_changes": 0,
                        "unique_users": 0,
                    },
                )
            )
            stack.enter_context(patch("orm.logging_functions.get_activity_over_time", return_value=[]))
            stack.enter_context(patch("orm.logging_functions.get_activity_by_type", return_value=[]))
            stack.enter_context(
                patch(
                    "orm.logging_functions.get_user_activity_page",
                    return_value={"items": [], "total": 0},
                )
            )

            admin_audit.render(30)

        assert (len(q1), len(a1), len(u1)) == (0, 1, 0), (
            f"Rerun 1 must open exactly one Admin Actions dialog; got Q={len(q1)} A={len(a1)} U={len(u1)}"
        )

        # ---- Rerun 2 -----------------------------------------------------
        # If the guard didn't reset, this would silently drop the dialog.
        q2: list = []
        a2: list = []
        u2: list = []

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(admin_audit, "st", stub))
            stack.enter_context(patch.object(admin_analytics, "st", stub))
            stack.enter_context(patch.object(admin_analytics, "_render_audit_question_dialog", side_effect=q2.append))
            stack.enter_context(patch.object(admin_analytics, "_render_admin_action_dialog", side_effect=a2.append))
            stack.enter_context(patch.object(admin_analytics, "_render_user_activity_dialog", side_effect=u2.append))
            stack.enter_context(
                patch.object(
                    admin_analytics,
                    "_cached_audit_filter_options",
                    return_value={"usernames": [], "orgs": []},
                )
            )
            stack.enter_context(
                patch(
                    "orm.logging_functions.get_question_audit_page",
                    return_value={"items": [], "total": 0},
                )
            )
            stack.enter_context(patch("orm.functions.get_all_users", return_value=[]))
            stack.enter_context(
                patch(
                    "orm.logging_functions.get_admin_actions_page",
                    return_value={"items": [a_item_2], "total": 1},
                )
            )
            stack.enter_context(
                patch(
                    "orm.logging_functions.get_admin_action_stats",
                    return_value={"total": 0, "user_changes": 0, "training_actions": 0, "failed": 0},
                )
            )
            stack.enter_context(patch("orm.logging_functions.get_admin_actions_by_type", return_value=[]))
            stack.enter_context(
                patch(
                    "orm.logging_functions.get_activity_stats",
                    return_value={
                        "logins_today": 0,
                        "failed_logins": 0,
                        "settings_changes": 0,
                        "unique_users": 0,
                    },
                )
            )
            stack.enter_context(patch("orm.logging_functions.get_activity_over_time", return_value=[]))
            stack.enter_context(patch("orm.logging_functions.get_activity_by_type", return_value=[]))
            stack.enter_context(
                patch(
                    "orm.logging_functions.get_user_activity_page",
                    return_value={"items": [], "total": 0},
                )
            )

            admin_audit.render(30)

        assert (len(q2), len(a2), len(u2)) == (0, 1, 0), (
            f"Rerun 2 must also open exactly one Admin Actions dialog (guard reset per rerun); "
            f"got Q={len(q2)} A={len(a2)} U={len(u2)}"
        )

    def test_render_resets_guard_before_tabs(self):
        """White-box check: even if a stale ``_audit_dialog_claimed_this_rerun``
        is True at entry (e.g. left over from the previous rerun's last
        execution), ``render`` must clear it before the tabs evaluate. This
        pins the second half of the fix (the guard reset in admin_audit.py).
        """
        from views import admin_audit

        a_item = _make_admin_action_item(id=1)

        stub = _SharedSessionStub(
            per_key_view_checked={"audit_actions_dataframe": [True]},
            initial_session={"_audit_dialog_claimed_this_rerun": True},
        )

        q_calls: list = []
        a_calls: list = []
        u_calls: list = []

        patchers = _patches_for_render(
            stub=stub,
            questions_items=[],
            actions_items=[a_item],
            activity_items=[],
            spy_q=q_calls.append,
            spy_a=a_calls.append,
            spy_u=u_calls.append,
        )

        import contextlib

        with contextlib.ExitStack() as stack:
            for p in patchers:
                stack.enter_context(p)
            admin_audit.render(30)

        # If the guard had not been reset, the Admin Actions dialog would
        # have been silently dropped.
        assert len(a_calls) == 1, (
            f"render() must reset _audit_dialog_claimed_this_rerun before tabs evaluate; "
            f"saw {len(a_calls)} A dialog call(s) (expected 1)"
        )


# ---------------------------------------------------------------------------
# (2) External-dialog collision: another tab opened a dialog before ours
# ---------------------------------------------------------------------------


class TestExternalDialogCollision:
    """Regression for: opening an audit dialog, clicking outside to dismiss, then
    going to Admin -> Users and clicking Export Users caused the Export dialog
    to flash and then be replaced by the audit dialog.

    Root cause: ``st.tabs`` evaluates every tab body on every rerun. ``admin.py``
    runs ``admin_users.render`` before ``admin_audit.render``. When the Users
    tab's button handler invokes ``export_users_dialog()``, Streamlit sets
    ``script_run_ctx.has_dialog_opened = True`` (the single-dialog-per-script-run
    flag). The Export body then writes a ``USER_EXPORT`` row via
    ``log_admin_action``, which shifts the ``thrive_admin_action`` page so the
    sticky ``edited_rows[0][View] = True`` from the previous rerun now points
    at the new top row. The Admin Actions tab's per-tab ``open_id`` gate then
    sees a mismatch and tries to fire ``_render_admin_action_dialog`` — but
    Streamlit allows only one dialog per run, so it raises and the frontend
    ends up showing the wrong dialog.

    Fix: ``admin_audit.render`` checks ``has_dialog_opened`` and pre-claims
    ``_audit_dialog_claimed_this_rerun`` so every inner-tab gate skips
    auto-open. These tests pin that behavior.
    """

    @staticmethod
    def _patch_dialog_opened(monkeypatch_stack, *, opened: bool):
        """Patch get_script_run_ctx to return a stub ctx with has_dialog_opened set.

        Uses contextlib.ExitStack so each test can compose this with the other
        patchers it needs.
        """
        import contextlib
        from unittest.mock import MagicMock, patch

        ctx_stub = MagicMock()
        ctx_stub.has_dialog_opened = opened

        # The fix imports get_script_run_ctx inside the try block at render-time,
        # so patch the source module — that import resolves dynamically per call.
        cm = patch(
            "streamlit.runtime.scriptrunner_utils.script_run_context.get_script_run_ctx",
            return_value=ctx_stub,
        )
        return cm if not isinstance(monkeypatch_stack, contextlib.ExitStack) else monkeypatch_stack.enter_context(cm)

    def test_external_dialog_already_open_blocks_admin_actions_auto_open(self):
        """Production repro: an earlier tab (Admin -> Users) already opened
        Export Users in this rerun, AND ``log_admin_action`` from that export
        shifted the Admin Actions audit table so the still-ticked checkbox
        now points to a different row. Without the fix, the audit code would
        try to call ``_render_admin_action_dialog`` for the new row and
        Streamlit would raise. With the fix, the audit code sees
        ``has_dialog_opened == True`` and skips."""
        from views import admin_audit

        # Original row the user ticked (id=271). After Export logs USER_EXPORT,
        # this row is now at index 1; index 0 is the new USER_EXPORT entry (id=999).
        new_top_after_shift = _make_admin_action_item(id=999)
        original_ticked = _make_admin_action_item(id=271)

        stub = _SharedSessionStub(
            # The checkbox at row 0 is still sticky-True from the prior rerun
            # when the user ticked the (then-top) original row.
            per_key_view_checked={"audit_actions_dataframe": [True, False]},
            # The gate already records that we opened the dialog for id=271.
            initial_session={"audit_actions_dialog_open_id": 271},
        )

        q_calls: list = []
        a_calls: list = []
        u_calls: list = []

        patchers = _patches_for_render(
            stub=stub,
            questions_items=[],
            # Items shifted: new USER_EXPORT row at index 0, original at index 1.
            actions_items=[new_top_after_shift, original_ticked],
            activity_items=[],
            spy_q=q_calls.append,
            spy_a=a_calls.append,
            spy_u=u_calls.append,
        )

        import contextlib
        from unittest.mock import MagicMock, patch

        ctx_stub = MagicMock()
        ctx_stub.has_dialog_opened = True

        with contextlib.ExitStack() as stack:
            for p in patchers:
                stack.enter_context(p)
            stack.enter_context(
                patch(
                    "streamlit.runtime.scriptrunner_utils.script_run_context.get_script_run_ctx",
                    return_value=ctx_stub,
                )
            )
            admin_audit.render(30)

        assert a_calls == [], (
            f"When script_run_ctx.has_dialog_opened is True, the Admin Actions "
            f"auto-open gate MUST skip — otherwise Streamlit raises and the "
            f"frontend shows the audit dialog over the external one. "
            f"Saw {len(a_calls)} call(s)."
        )
        # And no other audit tab should fire either.
        assert q_calls == []
        assert u_calls == []

    def test_external_dialog_already_open_blocks_questions_auto_open(self):
        """Same scenario but the user originally ticked a row on the Questions
        tab. Even without items shifting (Questions table is more stable —
        clicks don't log question audit entries), the guard should still skip
        any dialog open if Streamlit reports a dialog has already been
        claimed in this script run."""
        from views import admin_audit

        q_item = _make_question_item(user_message_id=314)

        stub = _SharedSessionStub(
            per_key_view_checked={"audit_dataframe": [True]},
            # Notably, open_id does NOT match the currently-ticked row — this
            # simulates the items-shift case for Questions too (e.g., a new
            # question logged by a concurrent user). Without the fix, the
            # gate would try to fire and collide with the external dialog.
            initial_session={"audit_dialog_open_user_message_id": 999},
        )

        q_calls: list = []
        a_calls: list = []
        u_calls: list = []

        patchers = _patches_for_render(
            stub=stub,
            questions_items=[q_item],
            actions_items=[],
            activity_items=[],
            spy_q=q_calls.append,
            spy_a=a_calls.append,
            spy_u=u_calls.append,
        )

        import contextlib
        from unittest.mock import MagicMock, patch

        ctx_stub = MagicMock()
        ctx_stub.has_dialog_opened = True

        with contextlib.ExitStack() as stack:
            for p in patchers:
                stack.enter_context(p)
            stack.enter_context(
                patch(
                    "streamlit.runtime.scriptrunner_utils.script_run_context.get_script_run_ctx",
                    return_value=ctx_stub,
                )
            )
            admin_audit.render(30)

        assert q_calls == [], (
            f"Questions auto-open must defer to an already-claimed dialog slot; saw {len(q_calls)} Q dialog call(s)"
        )
        assert a_calls == []
        assert u_calls == []

    def test_no_external_dialog_preserves_normal_auto_open(self):
        """Sanity check: when no other dialog has claimed the slot, the audit
        gate still fires normally. Pins that the new guard ONLY suppresses
        when ``has_dialog_opened`` is True — it doesn't silently break the
        happy path."""
        from views import admin_audit

        a_item = _make_admin_action_item(id=42)

        stub = _SharedSessionStub(
            per_key_view_checked={"audit_actions_dataframe": [True]},
        )

        q_calls: list = []
        a_calls: list = []
        u_calls: list = []

        patchers = _patches_for_render(
            stub=stub,
            questions_items=[],
            actions_items=[a_item],
            activity_items=[],
            spy_q=q_calls.append,
            spy_a=a_calls.append,
            spy_u=u_calls.append,
        )

        import contextlib
        from unittest.mock import MagicMock, patch

        ctx_stub = MagicMock()
        ctx_stub.has_dialog_opened = False

        with contextlib.ExitStack() as stack:
            for p in patchers:
                stack.enter_context(p)
            stack.enter_context(
                patch(
                    "streamlit.runtime.scriptrunner_utils.script_run_context.get_script_run_ctx",
                    return_value=ctx_stub,
                )
            )
            admin_audit.render(30)

        assert len(a_calls) == 1, (
            f"With no external dialog claimed, Admin Actions auto-open must still fire exactly once; saw {len(a_calls)}"
        )

    def test_missing_script_run_ctx_does_not_crash(self):
        """Defensive: outside a Streamlit script run (e.g., direct invocation
        in a test runner) ``get_script_run_ctx()`` returns None. The guard
        must treat that as "no claim" and not crash."""
        from views import admin_audit

        a_item = _make_admin_action_item(id=7)

        stub = _SharedSessionStub(
            per_key_view_checked={"audit_actions_dataframe": [True]},
        )

        q_calls: list = []
        a_calls: list = []
        u_calls: list = []

        patchers = _patches_for_render(
            stub=stub,
            questions_items=[],
            actions_items=[a_item],
            activity_items=[],
            spy_q=q_calls.append,
            spy_a=a_calls.append,
            spy_u=u_calls.append,
        )

        import contextlib
        from unittest.mock import patch

        with contextlib.ExitStack() as stack:
            for p in patchers:
                stack.enter_context(p)
            stack.enter_context(
                patch(
                    "streamlit.runtime.scriptrunner_utils.script_run_context.get_script_run_ctx",
                    return_value=None,
                )
            )
            admin_audit.render(30)

        # With ctx=None the guard treats it as "no external dialog", so the
        # normal auto-open path runs.
        assert len(a_calls) == 1
