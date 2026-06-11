"""Regression tests for the Admin → Audit umbrella tab dialog collision (#166).

Background
----------
``views/admin_audit.py:render`` builds an inner ``st.tabs`` with three child
audit tabs (Questions, Admin Actions, User Activity). Epic #169 / #170
swapped each tab's trigger primitive from
``st.dataframe(on_select=..., selection_mode='single-row')`` to
``st.data_editor`` + a leading labeled ``View`` ``CheckboxColumn`` that
auto-opens the detail dialog when exactly one row is ticked.
``st.tabs`` still evaluates *every* tab body on every rerun, and
Streamlit still forbids more than one ``st.dialog`` call per script run
— so the cross-tab claim guard remains as defense in depth.

The per-rerun guard, ``_audit_dialog_claimed_this_rerun``:

* ``views/admin_audit.py:render`` resets the flag to ``False`` at the top of
  every rerun, before ``st.tabs`` is created.
* Each of the three tab dialog branches checks-and-sets the flag *inside*
  the auto-open-on-tick branch, so at most one tab opens a dialog per
  rerun.

These tests pin both behaviours under the new (data_editor + auto-open)
flow.
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
        the Questions tab's ``View`` checkbox on a different row each
        time must each fire exactly one dialog. (If the guard leaked
        across reruns, the second rerun would silently drop the dialog.)
        """
        from views import admin_audit

        # Two distinct Questions items across the two reruns. Same
        # data_editor key, different ``user_message_id``s — so the
        # per-tab ``open_id`` gate naturally allows the second rerun's
        # dialog to fire (different id → different gate value).
        q_item_1 = _make_question_item(user_message_id=111)
        q_item_2 = _make_question_item(user_message_id=222)

        # A single shared stub across both reruns so session_state persists,
        # just like real Streamlit. Row 0 is ticked in both reruns; the
        # second rerun's row has a different ``user_message_id``.
        stub = _SharedSessionStub(
            per_key_view_checked={"audit_dataframe": [True]},
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
                    return_value={"items": [q_item_1], "total": 1},
                )
            )
            stack.enter_context(patch("orm.functions.get_all_users", return_value=[]))
            stack.enter_context(
                patch(
                    "orm.logging_functions.get_admin_actions_page",
                    return_value={"items": [], "total": 0},
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

        assert (len(q1), len(a1), len(u1)) == (1, 0, 0), (
            f"Rerun 1 must open exactly one Questions dialog; got Q={len(q1)} A={len(a1)} U={len(u1)}"
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
                    return_value={"items": [q_item_2], "total": 1},
                )
            )
            stack.enter_context(patch("orm.functions.get_all_users", return_value=[]))
            stack.enter_context(
                patch(
                    "orm.logging_functions.get_admin_actions_page",
                    return_value={"items": [], "total": 0},
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

        assert (len(q2), len(a2), len(u2)) == (1, 0, 0), (
            f"Rerun 2 must also open exactly one Questions dialog (guard reset per rerun); "
            f"got Q={len(q2)} A={len(a2)} U={len(u2)}"
        )

    def test_render_resets_guard_before_tabs(self):
        """White-box check: even if a stale ``_audit_dialog_claimed_this_rerun``
        is True at entry (e.g. left over from the previous rerun's last
        execution), ``render`` must clear it before the tabs evaluate. This
        pins the second half of the fix (the guard reset in admin_audit.py).
        """
        from views import admin_audit

        q_item = _make_question_item(user_message_id=1)

        stub = _SharedSessionStub(
            per_key_view_checked={"audit_dataframe": [True]},
            initial_session={"_audit_dialog_claimed_this_rerun": True},
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

        # If the guard had not been reset, the Questions dialog would have
        # been silently dropped.
        assert len(q_calls) == 1, (
            f"render() must reset _audit_dialog_claimed_this_rerun before tabs evaluate; "
            f"saw {len(q_calls)} Q dialog call(s) (expected 1)"
        )
