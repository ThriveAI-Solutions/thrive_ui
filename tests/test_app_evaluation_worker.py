"""app.py starts the evaluation worker exactly once, and only after a
successful database bootstrap."""

import runpy
import sys

import pytest

import evals.worker as worker_mod
import orm.models as orm_models


class _Stop(Exception):
    pass


class _SessionState(dict):
    """Supports both attribute access (st.session_state.cookies = ...) and the
    mapping .get() app.py uses."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class _FakeSt:
    def __init__(self):
        self.session_state = _SessionState()
        self.secrets = {"cookie": {"password": "x", "prefix": "t"}}
        self.errors = []

    def set_page_config(self, *a, **k):
        return None

    def cache_resource(self, func):
        # Passthrough so decorated bodies actually execute when called.
        return func

    def error(self, msg, *a, **k):
        self.errors.append(msg)

    def stop(self, *a, **k):
        raise _Stop

    # Reached only on the success path (we stop at cookie-not-ready before nav).
    def navigation(self, *a, **k):
        raise AssertionError("navigation reached; expected stop at cookie readiness")


class _FakeCookieManager:
    def __init__(self, *a, **k):
        pass

    def ready(self):
        return False  # halt app.py via st.stop() before navigation/auth


@pytest.fixture
def app_env(monkeypatch):
    fake_st = _FakeSt()
    fake_cookie_mod = type(sys)("streamlit_cookies_manager_ext")
    fake_cookie_mod.EncryptedCookieManager = _FakeCookieManager
    monkeypatch.setitem(sys.modules, "streamlit", fake_st)
    monkeypatch.setitem(sys.modules, "streamlit_cookies_manager_ext", fake_cookie_mod)

    started = {"count": 0}
    monkeypatch.setattr(worker_mod, "start_evaluation_worker", lambda *a, **k: started.__setitem__("count", started["count"] + 1))
    return fake_st, started, monkeypatch


def test_worker_starts_once_after_successful_bootstrap(app_env):
    fake_st, started, monkeypatch = app_env
    monkeypatch.setattr(orm_models, "init_db", lambda: None)

    with pytest.raises(_Stop):  # halts at cookie-not-ready, after worker start
        runpy.run_path("app.py", run_name="app_worker_test")

    assert started["count"] == 1
    assert fake_st.errors == []


def test_worker_not_started_when_bootstrap_fails(app_env):
    fake_st, started, monkeypatch = app_env

    def _boom():
        raise RuntimeError("migration failed")

    monkeypatch.setattr(orm_models, "init_db", _boom)

    with pytest.raises(_Stop):  # halts at the bootstrap-error st.stop()
        runpy.run_path("app.py", run_name="app_worker_test")

    assert started["count"] == 0
    assert fake_st.errors  # a bootstrap error was surfaced
