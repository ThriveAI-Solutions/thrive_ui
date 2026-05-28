"""Tests for the asyncio bridge used in agent/runtime.py.

The bridge runs a single persistent event loop forever on a dedicated daemon
thread; callers on any thread submit coroutines with run_coroutine_threadsafe
and block on the result. This keeps pydantic-ai's httpx transports bound to one
live loop and lets concurrent Streamlit script threads share it without the
"this event loop is already running" collision that run_until_complete caused.
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


def test_run_async_handles_concurrent_calls_from_multiple_threads():
    """Regression for 'RuntimeError: this event loop is already running'.

    Two Streamlit script threads can drive the shared loop at the same
    time (two sessions, or a rerun firing while a long run is still
    unwinding). The old bridge used asyncio.get_running_loop(), which
    only sees a loop on the *current* thread, then called
    run_until_complete on the shared loop — which raises if that loop is
    already running on another thread. Concurrent calls must succeed.
    """
    import threading

    from agent.runtime import _run_async

    # Force overlap: both threads enter _run_async together and stay busy
    # for the same window.
    barrier = threading.Barrier(2)
    results: dict[int, object] = {}
    errors: dict[int, BaseException] = {}

    async def work(tag: int) -> int:
        await asyncio.sleep(0.3)
        return tag

    def run(tag: int) -> None:
        barrier.wait()
        try:
            results[tag] = _run_async(work(tag))
        except BaseException as exc:  # noqa: BLE001 - capture for assertion
            errors[tag] = exc

    threads = [threading.Thread(target=run, args=(i,)) for i in (0, 1)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    assert results == {0: 0, 1: 1}


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
