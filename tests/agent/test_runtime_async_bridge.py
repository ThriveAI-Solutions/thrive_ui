"""Tests for the asyncio bridge used in agent/runtime.py.

Streamlit's script thread sometimes has a running event loop (Tornado),
sometimes does not. The bridge must:
- Detect a running loop and run the coroutine in a worker thread.
- Otherwise use a fresh loop (asyncio.run-equivalent).
"""

import asyncio

import pytest


def test_run_async_works_without_running_loop():
    from agent.runtime import _run_async

    async def coro():
        return 42

    assert _run_async(coro()) == 42


@pytest.mark.asyncio
async def test_run_async_works_inside_running_loop():
    """When a loop is already running, asyncio.run() raises
    'event loop already running'. _run_async must handle this."""
    from agent.runtime import _run_async

    async def coro():
        return "ok"

    # We're inside a running loop because pytest-asyncio gave us one.
    result = _run_async(coro())
    assert result == "ok"


@pytest.mark.asyncio
async def test_run_async_propagates_exceptions():
    from agent.runtime import _run_async

    class Boom(Exception):
        pass

    async def coro():
        raise Boom("scoped")

    with pytest.raises(Boom):
        _run_async(coro())


def test_run_async_reuses_a_persistent_loop_across_calls():
    """Pydantic AI's HTTP clients (httpx connection pools) bind their
    transports to the loop they were first used on. asyncio.run() per
    call would close that loop and leave the next call with dead
    transports — exactly the "TCPTransport closed" RuntimeError the
    user hit after clicking a patient. Two consecutive _run_async
    calls must share one loop."""
    import asyncio as _asyncio

    from agent.runtime import _run_async

    async def get_loop_id():
        return id(_asyncio.get_event_loop())

    a = _run_async(get_loop_id())
    b = _run_async(get_loop_id())
    assert a == b


def test_run_agentic_flow_closes_sqlite_session_on_exception(monkeypatch):
    """If the agent raises mid-run, the per-request SQLite session must
    still be closed — otherwise long-running Streamlit processes leak
    connections.
    """
    import agent.runtime as runtime_mod

    class _FakeSession:
        def __init__(self):
            self.committed = False
            self.closed = False

        def add(self, *a, **kw):
            pass

        def flush(self):
            pass

        def commit(self):
            self.committed = True

        def close(self):
            self.closed = True

    fake_session = _FakeSession()

    monkeypatch.setattr(runtime_mod, "SessionLocal", lambda: fake_session)
    monkeypatch.setattr(runtime_mod, "build_agent_deps", lambda s: object())

    class _BoomRunner:
        async def stream(self, *a, **kw):
            raise RuntimeError("simulated agent failure")
            yield  # unreachable, makes this an async generator

    monkeypatch.setattr(runtime_mod, "_runner", lambda: _BoomRunner())
    # Avoid touching st.session_state in add_message
    monkeypatch.setattr("utils.chat_bot_helper.add_message", lambda *a, **kw: None, raising=False)

    with pytest.raises(RuntimeError, match="simulated agent failure"):
        runtime_mod.run_agentic_message_flow("hi")

    assert fake_session.closed is True
    assert fake_session.committed is False  # commit skipped due to exception
