"""Agentic eval harness — runs the question×patient matrix against the
live warehouse and writes evals/results/<run_id>.json incrementally.

Usage:
    uv run python scripts/agent_eval_harness.py --dry-run
    uv run python scripts/agent_eval_harness.py --suggest-patients
    uv run python scripts/agent_eval_harness.py                # full run
    uv run python scripts/agent_eval_harness.py --only Q1 Q4 --skip-judge

Then: uv run python scripts/generate_eval_report.py evals/results/<run_id>.json

Requires .streamlit/secrets.toml with [ai_keys], [analytics_db], [agent].
Deps pattern follows scripts/agent_replay.py (headless, stub RAG fallback).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# sys.path shim — same pattern as scripts/agent_replay.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.db.analytics_adapter import AnalyticsDbAdapter
from agent.deps import AgentDeps
from agent.observability import configure_observability
from agent.runner import AgenticRunner
from evals.collect import run_turn
from evals.discovery import format_roster_snippet, suggest_patients
from evals.judge import build_judge, judge_turn
from evals.matrix import PlannedConversation, build_matrix, load_questions, load_roster
from evals.patients import resolve_patient

_REPO_ROOT = Path(__file__).resolve().parent.parent
_QUESTIONS = _REPO_ROOT / "evals/questions.yaml"
_ROSTER = _REPO_ROOT / "evals/roster.yaml"
_RESULTS_DIR = _REPO_ROOT / "evals/results"


class _NullRagAdapter:
    """Stub RAG when ChromaDB can't be opened — see scripts/agent_replay.py."""

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
        print(f"warning: ChromaDB unavailable ({type(exc).__name__}: {exc}); using stub RAG", flush=True)
        return _NullRagAdapter()


def _build_deps(adapter, rag, selected_patient, session_id: str) -> AgentDeps:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from orm.models import RoleTypeEnum

    return AgentDeps(
        user_id=0,
        user_role=RoleTypeEnum.DOCTOR,
        session_id=session_id,
        selected_patient=selected_patient,
        last_dataframe=None,
        last_sql=None,
        last_query_meta=None,
        analytics_db=adapter,
        rag=rag,
        sqlite_session=sessionmaker(bind=create_engine("sqlite:///:memory:"))(),
        run_logger=None,
    )


def _model_info() -> dict:
    import streamlit as st

    ai_keys = st.secrets.get("ai_keys", {})
    if ai_keys.get("ollama_model"):
        inferred = "ollama"
    elif ai_keys.get("anthropic_model"):
        inferred = "anthropic"
    elif ai_keys.get("bedrock_model_id"):
        inferred = "bedrock"
    else:
        inferred = ""
    provider = ai_keys.get("provider") or inferred
    model = ai_keys.get("ollama_model") or ai_keys.get("anthropic_model") or ai_keys.get("bedrock_model_id") or ""
    return {"provider": provider, "model": model}


def _write_results(path: Path, results: dict) -> None:
    """Atomic incremental write — crash-safe at conversation granularity."""
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(results, indent=2, default=str))
    os.replace(tmp, path)


async def _run_conversation(
    runner: AgenticRunner,
    adapter: AnalyticsDbAdapter,
    rag,
    planned: PlannedConversation,
    judge,
    run_id: str,
) -> dict:
    record = {
        "conversation_id": planned.conversation_id,
        "question_id": planned.question_id,
        "question_title": planned.question_title,
        "reviewer_note": planned.reviewer_note,
        "patient": {"source_id": planned.source_id, "display_name": "", "label": planned.patient_label},
        "status": "ok",
        "error": None,
        "turns": [],
    }
    try:
        selected = resolve_patient(adapter, planned.source_id)
        # Clears the lookup query AND any stale entries left if a prior
        # conversation errored mid-turn, so turn 1 attribution starts clean.
        adapter.pop_sql_log()
        record["patient"]["display_name"] = selected.display_name
        deps = _build_deps(adapter, rag, selected, session_id=f"{run_id}-{planned.conversation_id}")

        message_history = None
        for index, planned_turn in enumerate(planned.turns):
            turn, all_messages = await run_turn(runner, deps, planned_turn.prompt, message_history=message_history)
            turn["index"] = index
            turn["role"] = planned_turn.role
            if judge is not None:
                summaries = [
                    f"{tc['tool_name']}: {tc.get('result_summary', '')}"
                    for tc in turn["tool_calls"]
                    if tc.get("completed")
                ]
                turn["judge"] = await judge_turn(judge, planned_turn.prompt, turn["answer"], summaries)
            else:
                turn["judge"] = None
            record["turns"].append(turn)
            if all_messages:
                message_history = all_messages
    except Exception as exc:
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=_QUESTIONS)
    parser.add_argument("--roster", type=Path, default=_ROSTER)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--only", nargs="*", default=None, help="Subset of question IDs")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved matrix; no LLM/DB")
    parser.add_argument("--suggest-patients", action="store_true", help="Print roster candidates from the warehouse")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--limit-patients", type=int, default=None, help="Run only the first N roster patients")
    args = parser.parse_args()

    questions = load_questions(args.questions)

    if args.suggest_patients:
        defaults, _ = (
            load_roster(args.roster)
            if args.roster.exists()
            else ({"date_start": "2024-01-01", "date_end": datetime.now().date().isoformat()}, [])
        )
        adapter = AnalyticsDbAdapter.from_streamlit_secrets()
        suggestions = suggest_patients(adapter, defaults["date_start"], defaults["date_end"])
        print(format_roster_snippet(suggestions))
        return 0

    if not args.roster.exists():
        print(f"error: {args.roster} not found — copy evals/roster.example.yaml and fill in source_ids")
        return 2
    defaults, patients = load_roster(args.roster)
    if args.limit_patients is not None:
        patients = patients[: args.limit_patients]
    matrix = build_matrix(questions, defaults, patients, only=args.only)

    if args.dry_run:
        print(f"{len(matrix)} conversations, {sum(len(c.turns) for c in matrix)} turns:\n")
        for convo in matrix:
            print(f"[{convo.conversation_id}]")
            for turn in convo.turns:
                print(f"  ({turn.role}) {turn.prompt}")
        return 0

    # --out is overwrite-only (no resume); when given, its stem becomes the
    # run_id so the filename and the embedded id can't disagree.
    if args.out is not None:
        out_path = args.out
        run_id = out_path.stem
    else:
        run_id = f"eval-{datetime.now():%Y%m%d-%H%M%S}"
        out_path = _RESULTS_DIR / f"{run_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        configure_observability()
        adapter = AnalyticsDbAdapter.from_streamlit_secrets()
        rag = _build_rag()
        runner = AgenticRunner()
        judge = None if args.skip_judge else build_judge()
    except Exception as exc:
        print(
            f"error: failed to initialize ({type(exc).__name__}: {exc}) — "
            f"check [ai_keys], [analytics_db], [agent] in .streamlit/secrets.toml"
        )
        return 2

    results = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": _model_info(),
        "defaults": defaults,
        "conversations": [],
    }
    _write_results(out_path, results)

    loop = asyncio.new_event_loop()
    try:
        for i, convo in enumerate(matrix, 1):
            print(f"[{i}/{len(matrix)}] {convo.conversation_id} ...", flush=True)
            record = loop.run_until_complete(_run_conversation(runner, adapter, rag, convo, judge, run_id))
            results["conversations"].append(record)
            _write_results(out_path, results)
            status = record["status"] if record["status"] != "ok" else f"{len(record['turns'])} turns"
            print(f"    -> {status}", flush=True)
    finally:
        loop.close()

    errors = sum(1 for c in results["conversations"] if c["status"] == "error")
    print(f"\ndone: {len(results['conversations'])} conversations ({errors} errored)")
    print(f"results: {out_path}")
    print(f"next:    uv run python scripts/generate_eval_report.py {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
