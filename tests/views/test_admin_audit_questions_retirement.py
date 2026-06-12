"""Phase 4 — Questions tab retirement + deep-link migration (Epic #190).

Asserts:
  * The Audit umbrella no longer renders a "Questions" tab.
  * The legacy ``audit_trail_pref_user_id`` deep-link contract is now
    honoured by the Queries tab (admins navigating from Manage Users →
    "View question audit for <username>" still land with their user
    filter pre-populated).
  * The forward-compat ``audit_queries_pref_user_id`` key works the same
    way (new surfaces can adopt the new name without breaking).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from orm.models import RoleTypeEnum


# ---------------------------------------------------------------------------
# (1) Umbrella tab list no longer includes "Questions"
# ---------------------------------------------------------------------------


def test_audit_umbrella_drops_questions_tab():
    from views import admin_audit

    captured_tab_labels: list[list[str]] = []

    class _Stub:
        def __init__(self):
            self.session_state = {}
            self.secrets = {"agent_logging": {"mode": "full"}}

        def tabs(self, labels):
            captured_tab_labels.append(list(labels))
            # Each "tab" is a context manager that swallows whatever the body does.
            return [MagicMock() for _ in labels]

    stub = _Stub()
    # Stub out the body renderers so this test only asserts the tab labels.
    with (
        patch.object(admin_audit, "st", stub),
        patch.object(admin_audit, "admin_audit_queries", MagicMock()),
        patch.object(admin_audit, "admin_audit_by_patient", MagicMock()),
        patch.object(admin_audit, "_render_audit_tab", MagicMock()),
        patch.object(admin_audit, "_render_activity_tab", MagicMock()),
    ):
        admin_audit.render(30)

    assert captured_tab_labels, "render() must call st.tabs"
    assert captured_tab_labels[0] == ["Queries", "By Patient", "Admin Actions", "User Activity"]
    assert "Questions" not in captured_tab_labels[0]


# ---------------------------------------------------------------------------
# (2) + (3) Deep-link prefill — both the legacy and forward-compat keys
# ---------------------------------------------------------------------------


def _make_prefill_stub(*, initial_session_state=None):
    captured_multiselect: list[dict] = []

    class _Stub:
        def __init__(self):
            self.session_state = dict(initial_session_state or {})
            self.session_state.setdefault("user_role", RoleTypeEnum.ADMIN.value)
            self.secrets = {"agent_logging": {"mode": "full"}}
            self.column_config = MagicMock()

        def multiselect(self, label, options=None, key=None, **kw):
            captured_multiselect.append({"label": label, "options": list(options or []), "key": key, "kwargs": kw})
            # If a previous code path (deep-link prefill) wrote to session_state
            # for this key, respect that value. Otherwise return [].
            val = self.session_state.get(key, [])
            self.session_state.setdefault(key, val)
            return val

        def text_input(self, *_a, key=None, **_kw):
            self.session_state.setdefault(key, "")
            return ""

        def selectbox(self, *_a, options=None, index=0, key=None, **_kw):
            opts = list(options or [])
            val = opts[index] if opts else None
            self.session_state.setdefault(key, val)
            return val

        def number_input(self, *_a, key=None, **_kw):
            self.session_state.setdefault(key, 1)
            return 1

        def radio(self, *_a, options=None, index=0, key=None, **_kw):
            opts = list(options or [])
            val = opts[index] if opts else None
            self.session_state.setdefault(key, val)
            return val

        def columns(self, spec):
            n = spec if isinstance(spec, int) else len(spec)
            return [MagicMock() for _ in range(n)]

        def data_editor(self, df, **_kw):
            return df

        def dataframe(self, *_a, **_kw):
            return MagicMock()

        def expander(self, *_a, **_kw):
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

        def divider(self):
            pass

        def warning(self, *_a, **_kw):
            pass

        def error(self, *_a, **_kw):
            pass

        def download_button(self, *_a, **_kw):
            pass

        def code(self, *_a, **_kw):
            pass

        def rerun(self):
            pass

        def dialog(self, _title):
            def _decorator(fn):
                return fn

            return _decorator

    stub = _Stub()
    return stub, {"multiselect": captured_multiselect}


def test_audit_trail_pref_user_id_prefills_queries_user_filter():
    """Legacy deep-link key (set by chat_bot.py / admin_users.py) must
    pre-populate the Queries tab's user filter."""
    from views import admin_audit_queries

    stub, _ = _make_prefill_stub(initial_session_state={"audit_trail_pref_user_id": 7})

    with (
        patch.object(admin_audit_queries, "st", stub),
        patch.object(
            admin_audit_queries,
            "_cached_audit_filter_options",
            return_value={"usernames": ["alice"], "orgs": []},
        ),
        patch(
            "orm.functions.get_all_users",
            return_value=[{"id": 7, "username": "alice"}, {"id": 8, "username": "bob"}],
        ),
        patch.object(
            admin_audit_queries,
            "get_per_query_audit_page",
            return_value={"items": [], "total": 0},
        ),
    ):
        admin_audit_queries._render_queries_tab(30)

    assert stub.session_state.get("queries_user_filter") == ["alice"]
    # The prefill key is consumed via .pop() so a rerun doesn't re-apply it.
    assert "audit_trail_pref_user_id" not in stub.session_state


def test_audit_queries_pref_user_id_also_accepted():
    """The forward-compat key (new surfaces can adopt this name) works the
    same way as the legacy key."""
    from views import admin_audit_queries

    stub, _ = _make_prefill_stub(initial_session_state={"audit_queries_pref_user_id": 8})

    with (
        patch.object(admin_audit_queries, "st", stub),
        patch.object(
            admin_audit_queries,
            "_cached_audit_filter_options",
            return_value={"usernames": ["alice", "bob"], "orgs": []},
        ),
        patch(
            "orm.functions.get_all_users",
            return_value=[{"id": 7, "username": "alice"}, {"id": 8, "username": "bob"}],
        ),
        patch.object(
            admin_audit_queries,
            "get_per_query_audit_page",
            return_value={"items": [], "total": 0},
        ),
    ):
        admin_audit_queries._render_queries_tab(30)

    assert stub.session_state.get("queries_user_filter") == ["bob"]
    assert "audit_queries_pref_user_id" not in stub.session_state
