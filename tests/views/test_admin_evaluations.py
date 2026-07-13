"""Admin Evaluations launch dispatch, guards, and snapshot handling."""

import types
from unittest.mock import MagicMock

import pytest

import views.admin_evaluations as ae


class _Stop(Exception):
    pass


class _Rerun(Exception):
    pass


@pytest.fixture
def fake_st(monkeypatch):
    st = MagicMock()
    st.session_state = {}
    st.stop.side_effect = _Stop
    st.rerun.side_effect = _Rerun
    monkeypatch.setattr(ae, "st", st)
    return st


def test_guard_blocks_non_admin(fake_st):
    fake_st.session_state["user_role"] = 1  # doctor
    with pytest.raises(_Stop):
        ae._guard_admin()


def test_guard_allows_admin(fake_st):
    fake_st.session_state["user_role"] = 0
    ae._guard_admin()  # no raise


def test_admin_id_reads_json_cookie(fake_st):
    # Local auth stores user_id in the cookie manager as a JSON string, not
    # directly in session_state — _admin_id must resolve it or every
    # service call fails admin authorization.
    fake_st.session_state["cookies"] = {"user_id": "16"}
    assert ae._admin_id() == 16


def test_admin_id_prefers_direct_session_value(fake_st):
    fake_st.session_state["user_id"] = 7
    fake_st.session_state["cookies"] = {"user_id": "16"}
    assert ae._admin_id() == 7


def test_admin_id_defaults_to_zero_when_absent(fake_st):
    assert ae._admin_id() == 0


def test_single_case_launch_is_synchronous_and_routes_to_report(fake_st, monkeypatch):
    view = types.SimpleNamespace(run_id="r1", execution_mode="synchronous", status="completed", total_cases=1)
    calls = {}

    def _fake_launch(ids, aid):
        calls["ids"] = ids
        return view

    monkeypatch.setattr(ae, "launch_evaluation", _fake_launch)
    with pytest.raises(_Rerun):
        ae._launch([5], admin_id=1)
    assert calls["ids"] == [5]
    assert fake_st.session_state["evaluation_run_id"] == "r1"


def test_multi_case_launch_is_async_toast(fake_st, monkeypatch):
    view = types.SimpleNamespace(run_id="r2", execution_mode="asynchronous", status="queued", total_cases=3)
    monkeypatch.setattr(ae, "launch_evaluation", lambda ids, aid: view)
    ae._launch([5, 6, 7], admin_id=1)
    fake_st.toast.assert_called_once()
    assert "evaluation_run_id" not in fake_st.session_state


def test_empty_launch_warns(fake_st, monkeypatch):
    called = {"n": 0}
    monkeypatch.setattr(ae, "launch_evaluation", lambda ids, aid: called.__setitem__("n", called["n"] + 1))
    ae._launch([], admin_id=1)
    fake_st.warning.assert_called_once()
    assert called["n"] == 0


def test_snapshot_collects_ids_and_handles_unavailable(fake_st, monkeypatch):
    good = types.SimpleNamespace(id=100)

    def snap(fid, aid):
        if fid == 2:
            raise ae.SnapshotUnavailable("logging disabled")
        return good

    monkeypatch.setattr(ae, "snapshot_feedback_case", snap)
    ids = ae._snapshot_feedback_ids([1, 2], admin_id=1)
    assert ids == [100]
    fake_st.warning.assert_called_once()
