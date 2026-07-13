"""The Admin umbrella page wires all six sub-tabs, including Evaluations."""

import runpy
import sys


import views.admin_analytics as admin_analytics
from views import (
    admin_analytics_consolidated,
    admin_audit,
    admin_evaluations,
    admin_feedback,
    admin_training,
    admin_users,
)


class _Ctx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeSt:
    def __init__(self):
        self.session_state = {"user_role": 0}  # admin

    def title(self, *a, **k):
        return None

    def columns(self, spec, **k):
        n = spec if isinstance(spec, int) else len(spec)
        return [_Ctx() for _ in range(n)]

    def tabs(self, labels, **k):
        self.tab_labels = list(labels)
        return [_Ctx() for _ in labels]

    def segmented_control(self, *a, **k):
        return "30 days"

    def button(self, *a, **k):
        return False

    def error(self, *a, **k):
        return None

    def stop(self, *a, **k):
        raise AssertionError("admin guard should not stop for an admin user")


def test_admin_page_renders_evaluations_tab(monkeypatch):
    fake = _FakeSt()
    calls = {}

    for name, mod in {
        "users": admin_users,
        "training": admin_training,
        "analytics": admin_analytics_consolidated,
        "audit": admin_audit,
        "feedback": admin_feedback,
        "evaluations": admin_evaluations,
    }.items():
        monkeypatch.setattr(mod, "render", (lambda name: lambda days: calls.__setitem__(name, days))(name))

    # _guard_admin lives in admin_analytics and reads its own module-level st.
    monkeypatch.setattr(admin_analytics, "st", fake)
    monkeypatch.setitem(sys.modules, "streamlit", fake)

    runpy.run_path("views/admin.py", run_name="views.admin")

    assert "Evaluations" in fake.tab_labels
    assert calls["evaluations"] == 30
    # Every sub-tab received the shared days_int.
    assert set(calls) == {"users", "training", "analytics", "audit", "feedback", "evaluations"}
