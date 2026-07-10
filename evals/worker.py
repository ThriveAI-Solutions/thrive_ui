"""Process-local evaluation worker: durable queue, heartbeat, recovery.

A single daemon thread claims queued *asynchronous* evaluation runs one at a
time, executes them through :func:`orm.evaluation_functions.run_evaluation_cases`,
heartbeats while working, and requeues runs abandoned by a dead worker.

The claim is serialized in SQLite with ``BEGIN IMMEDIATE`` and refuses to start
a second concurrent run, so the "one active asynchronous run per deployment"
invariant holds even if more than one app process races. The
:class:`EvaluationQueue` protocol keeps the claim/heartbeat/finish contract
swappable for a future external (non-SQLite) queue without touching callers.

Synchronous single-feedback runs never reach this worker — they run inline in
the Streamlit request via
:func:`orm.evaluation_functions.execute_synchronous_run`.
"""

from __future__ import annotations

import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Optional, Protocol

from orm.models import engine
from utils.quick_logger import get_logger

logger = get_logger(__name__)

POLL_INTERVAL_S = 5.0
HEARTBEAT_TIMEOUT_S = 120.0
_RETENTION_INTERVAL_S = 24 * 60 * 60


class EvaluationQueue(Protocol):
    """Swappable claim/heartbeat/finish contract for durable runs."""

    def claim_next(self, worker_id: str) -> Optional[str]: ...

    def heartbeat(self, run_id: str, worker_id: str) -> None: ...

    def finish(self, run_id: str, status: str) -> None: ...


class SqliteEvaluationQueue:
    """SQLite-backed queue. Claims are serialized with ``BEGIN IMMEDIATE`` and
    gated on there being no other asynchronous run already ``running``."""

    def claim_next(self, worker_id: str) -> Optional[str]:
        raw = engine.raw_connection()
        try:
            cur = raw.cursor()
            cur.execute("BEGIN IMMEDIATE")
            # Enforce one active asynchronous run per deployment.
            cur.execute(
                "SELECT COUNT(*) FROM thrive_evaluation_run "
                "WHERE status = 'running' AND execution_mode = 'asynchronous'"
            )
            if cur.fetchone()[0] > 0:
                raw.rollback()
                return None
            cur.execute(
                "SELECT id, run_id FROM thrive_evaluation_run "
                "WHERE status = 'queued' AND execution_mode = 'asynchronous' "
                "ORDER BY created_at, id LIMIT 1"
            )
            row = cur.fetchone()
            if row is None:
                raw.rollback()
                return None
            run_pk, run_public = row
            cur.execute(
                "UPDATE thrive_evaluation_run "
                "SET status = 'running', started_at = CURRENT_TIMESTAMP, heartbeat_at = CURRENT_TIMESTAMP "
                "WHERE id = ?",
                (run_pk,),
            )
            raw.commit()
            return run_public
        except Exception:
            raw.rollback()
            raise
        finally:
            raw.close()

    def heartbeat(self, run_id: str, worker_id: str) -> None:
        raw = engine.raw_connection()
        try:
            cur = raw.cursor()
            cur.execute(
                "UPDATE thrive_evaluation_run SET heartbeat_at = CURRENT_TIMESTAMP WHERE run_id = ?",
                (run_id,),
            )
            raw.commit()
        finally:
            raw.close()

    def finish(self, run_id: str, status: str) -> None:
        # Terminal status is written by run_evaluation_cases' finalizer; this is
        # a no-op hook kept for protocol completeness and external backends.
        return None


def build_default_resources(skip_judge: bool = False):
    """Construct the real executor collaborators (warehouse adapter, RAG,
    agentic runner, judge) from Streamlit secrets. Mirrors the CLI harness so
    worker and CLI stay behaviorally identical."""
    from agent.db.analytics_adapter import AnalyticsDbAdapter
    from agent.observability import configure_observability
    from agent.runner import AgenticRunner
    from evals.executor import EvaluationResources
    from evals.judge import build_judge

    configure_observability()
    adapter = AnalyticsDbAdapter.from_streamlit_secrets()
    rag = _build_rag()
    runner = AgenticRunner()
    judge = None if skip_judge else build_judge()
    return EvaluationResources(runner=runner, adapter=adapter, rag=rag, judge=judge)


class _NullRagAdapter:
    """Stub RAG when ChromaDB can't be opened (mirrors the CLI harness)."""

    def search(self, query: str, kind: str | None = None, limit: int = 5):
        return []

    def upsert(self, *args, **kwargs):
        return None


def _build_rag():
    import streamlit as st

    try:
        import chromadb

        from agent.rag.chroma_adapter import ChromaRagAdapter

        chroma_path = st.secrets.get("rag_model", {}).get("chroma_path", "./chromadb")
        return ChromaRagAdapter(chromadb.PersistentClient(path=chroma_path))
    except Exception as exc:
        logger.warning("ChromaDB unavailable (%s); using stub RAG for evaluation worker", type(exc).__name__)
        return _NullRagAdapter()


def _process_one_run(run_id: str, queue: EvaluationQueue, worker_id: str) -> None:
    import asyncio

    from orm.evaluation_functions import run_evaluation_cases

    resources = build_default_resources()

    def _heartbeat(rid: str) -> None:
        queue.heartbeat(rid, worker_id)

    loop = asyncio.new_event_loop()
    try:
        view = loop.run_until_complete(run_evaluation_cases(run_id, resources, heartbeat=_heartbeat))
        queue.finish(run_id, view.status)
        logger.info("Evaluation run %s finished: %s", run_id, view.status)
    finally:
        loop.close()


def _worker_loop(queue: EvaluationQueue, worker_id: str, stop_event: threading.Event) -> None:
    from orm.evaluation_functions import mark_interrupted_runs, purge_expired_evaluation_payloads

    now = datetime.now(timezone.utc)
    try:
        purge_expired_evaluation_payloads(now)
    except Exception:
        logger.exception("Initial retention purge failed")
    last_retention = time.monotonic()

    while not stop_event.is_set():
        try:
            mark_interrupted_runs(datetime.now(timezone.utc), HEARTBEAT_TIMEOUT_S)
            run_id = queue.claim_next(worker_id)
            if run_id is not None:
                _process_one_run(run_id, queue, worker_id)
                continue  # drain the queue without sleeping between runs
            if time.monotonic() - last_retention >= _RETENTION_INTERVAL_S:
                purge_expired_evaluation_payloads(datetime.now(timezone.utc))
                last_retention = time.monotonic()
        except Exception:
            logger.exception("Evaluation worker iteration failed; continuing")
        stop_event.wait(POLL_INTERVAL_S)


_worker_lock = threading.Lock()
_worker_thread: Optional[threading.Thread] = None
_stop_event: Optional[threading.Event] = None


def start_evaluation_worker(queue: Optional[EvaluationQueue] = None) -> None:
    """Start the singleton daemon worker. Idempotent: Streamlit reruns cannot
    spawn a second thread. Call only after successful DB bootstrap."""
    global _worker_thread, _stop_event
    with _worker_lock:
        if _worker_thread is not None and _worker_thread.is_alive():
            return
        queue = queue or SqliteEvaluationQueue()
        worker_id = f"worker-{uuid.uuid4()}"
        _stop_event = threading.Event()
        _worker_thread = threading.Thread(
            target=_worker_loop,
            args=(queue, worker_id, _stop_event),
            name="evaluation-worker",
            daemon=True,
        )
        _worker_thread.start()
        logger.info("Started evaluation worker %s", worker_id)


def stop_evaluation_worker() -> None:
    """Signal the worker to stop (used in tests)."""
    global _worker_thread, _stop_event
    with _worker_lock:
        if _stop_event is not None:
            _stop_event.set()
        _worker_thread = None
        _stop_event = None
