"""CLI replay tool for the agentic chat path.

Drives the production agent runner against the configured Redshift +
ChromaDB + Ollama LLM and prints every streamed event to stdout. Lets us
iterate on prompts, tool descriptions, and SQL templates in seconds
without bouncing the Streamlit UI.

Reads `.streamlit/secrets.toml` the same way the live app does, so the
provider, model, RAG store, and warehouse credentials all match what
runs in the dev/prod streamlit. The only thing the CLI fakes is the
session-state plumbing (selected patient, sqlite ORM session) — those
are stubbed because a one-shot replay doesn't need persistence.

Usage from the project root (or thrive_ui_dev on the server):

    uv run python scripts/agent_replay.py "give me a list of people in zip 14223 with high blood pressure"
    uv run python scripts/agent_replay.py "..." --model gpt-oss:32b
    uv run python scripts/agent_replay.py "..." --user-role nurse

Exit codes: 0 if a final response was produced, 1 if no final, 2 if a cap
was hit. Pipes-friendly: ANSI colors auto-disable when stdout isn't a TTY.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path

# Make the project root importable when invoked directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chromadb  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from datetime import date, datetime  # noqa: E402

from agent.audit import AuditLogger  # noqa: E402
from agent.db.analytics_adapter import AnalyticsDbAdapter  # noqa: E402
from agent.deps import AgentDeps, SelectedPatient  # noqa: E402
from agent.rag.chroma_adapter import ChromaRagAdapter  # noqa: E402
from agent.runner import AgenticRunner  # noqa: E402
from agent.state import (  # noqa: E402
    AssistantTextCompletedEvent,
    AssistantTextDeltaEvent,
    CapReachedEvent,
    CohortSampleEvent,
    FinalResponseEvent,
    PatientChooserEvent,
    ThinkingCompletedEvent,
    ThinkingDeltaEvent,
    ToolCallCompleted,
    ToolCallStarted,
)
from orm.models import RoleTypeEnum  # noqa: E402


class C:
    """ANSI color escapes — overwritten with empty strings when stdout
    isn't a TTY (or --no-color is passed) so piped output stays clean."""

    RESET = "\x1b[0m"
    BOLD = "\x1b[1m"
    DIM = "\x1b[2m"
    CYAN = "\x1b[36m"
    BLUE = "\x1b[34m"
    GREEN = "\x1b[32m"
    YELLOW = "\x1b[33m"
    RED = "\x1b[31m"
    MAGENTA = "\x1b[35m"
    GRAY = "\x1b[90m"


def _disable_color() -> None:
    for attr in list(vars(C)):
        if attr.startswith("_") or attr == "__module__":
            continue
        setattr(C, attr, "")


def _build_model_override(model_name: str):
    """Build an OpenAI-compatible Ollama model with `model_name`,
    re-using the same provider + http client + reasoning settings as
    `agent.models.build_model` does for the default path."""
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
    from pydantic_ai.providers.openai import OpenAIProvider
    import streamlit as st

    from agent.models import _ollama_http_client

    ai_keys = st.secrets.get("ai_keys", {})
    host = ai_keys.get("ollama_host", "http://localhost:11434")
    agent_cfg = st.secrets.get("agent", {})
    per_model = agent_cfg.get("ollama_think_per_model", {}) or {}
    if model_name in per_model:
        think = bool(per_model[model_name])
    else:
        think = bool(agent_cfg.get("ollama_think", True))
    settings = OpenAIChatModelSettings(extra_body={"reasoning_effort": "high" if think else "none"})
    return OpenAIChatModel(
        model_name,
        provider=OpenAIProvider(
            base_url=f"{host.rstrip('/')}/v1",
            http_client=_ollama_http_client(),
        ),
        settings=settings,
    )


class _NullRagAdapter:
    """Stub RAG used when the real ChromaDB can't be opened (e.g. another
    process holds the SQLite file, or this user lacks write perms).
    `search_knowledge_base` will return an empty list — tools that depend
    on RAG retrieval will degrade, but the rest of the agent (system
    prompt routing, schema catalog injection via the run_sql prepare
    hook, all clinical-data tools) works unchanged."""

    def search(self, query: str, kind: str | None = None, limit: int = 5):
        return []

    def upsert(self, *args, **kwargs):
        return None


def _build_deps(role: str) -> AgentDeps:
    """Stub the session-state bits build_agent_deps reads from Streamlit;
    everything else (Redshift adapter, ChromaDB) uses the same factories
    the live app does so behavior matches.

    ChromaDB has a quirk: opening even a read-only PersistentClient
    requires write access to chroma.sqlite3 (it touches a lock file).
    On the dev server `streamlitadmin` can't write the streamlituser-
    owned file, so we fall back to a no-op stub. The agent's
    schema-context flow (the run_sql prepare hook) reads from the
    in-process RUN_SQL_EXAMPLES constant, not chromadb, so most
    prompt-iteration runs don't need real RAG. Pass `--require-rag` to
    fail loudly if the stub would be used."""
    import streamlit as st

    analytics_db = AnalyticsDbAdapter.from_streamlit_secrets()
    chroma_path = st.secrets.get("rag_model", {}).get("chroma_path", "./chromadb")
    try:
        rag = ChromaRagAdapter(chromadb.PersistentClient(path=chroma_path))
    except Exception as exc:
        print(
            f"{C.YELLOW}⚠  ChromaDB unavailable ({exc.__class__.__name__}: {exc}); "
            f"using no-op RAG stub. search_knowledge_base will return empty.{C.RESET}",
            flush=True,
        )
        rag = _NullRagAdapter()

    # In-memory ORM session — AuditLogger writes here are throwaway and
    # are tolerated to fail silently in the runner.
    sqlite_session = sessionmaker(bind=create_engine("sqlite:///:memory:"))()

    user_role = {
        "admin": RoleTypeEnum.ADMIN,
        "doctor": RoleTypeEnum.DOCTOR,
        "nurse": RoleTypeEnum.NURSE,
        "patient": RoleTypeEnum.PATIENT,
    }[role.lower()]
    session_id = f"cli-replay-{uuid.uuid4().hex[:8]}"

    return AgentDeps(
        user_id=0,
        user_role=user_role,
        session_id=session_id,
        selected_patient=None,
        last_dataframe=None,
        last_sql=None,
        last_query_meta=None,
        analytics_db=analytics_db,
        rag=rag,
        sqlite_session=sqlite_session,
        audit_logger=AuditLogger(
            session=sqlite_session,
            session_id=session_id,
            user_id=0,
            user_role=int(user_role.value),
        ),
    )


def _format_args(args: dict) -> str:
    try:
        compact = json.dumps(args, default=str, separators=(", ", ": "))
    except TypeError:
        compact = str(args)
    if len(compact) <= 140:
        return compact
    indented = json.dumps(args, default=str, indent=2)
    return "\n      " + indented.replace("\n", "\n      ")


def _format_sql_block(sql: str, params: dict | None) -> list[str]:
    lines = ["   sql:"]
    for line in sql.strip().splitlines():
        lines.append(f"   {C.DIM}  {line}{C.RESET}")
    if params:
        lines.append(f"   {C.DIM}  params: {params}{C.RESET}")
    return lines


def _auto_select_first(payload: dict) -> SelectedPatient | None:
    """Pick the first patient from a find_patient chooser payload, mimicking
    what the UI does when the user clicks the top row. Without this in the
    CLI, follow-up clinical-data turns refuse with 'no patient slot.'"""
    matches = payload.get("matches") or payload.get("sample") or []
    if not isinstance(matches, list) or not matches:
        return None
    first = matches[0]
    if not isinstance(first, dict):
        return None
    src = first.get("source_id")
    if not src:
        return None
    raw_dob = first.get("dob") or first.get("date_of_birth")
    parsed_dob = None
    if raw_dob:
        try:
            parsed_dob = date.fromisoformat(raw_dob) if isinstance(raw_dob, str) else raw_dob
        except (TypeError, ValueError):
            parsed_dob = None
    return SelectedPatient(
        source_id=src,
        display_name=first.get("display_name") or first.get("full_name") or "",
        dob=parsed_dob,
        selected_at=datetime.now(),
        selection_origin="agent_disambiguation",
    )


async def _run_one_turn(
    runner: AgenticRunner,
    question: str,
    deps: AgentDeps,
    message_history: list | None,
    auto_select: bool,
) -> tuple[object | None, str | None, list]:
    """Stream a single agent turn and print events. Returns
    (final_response, cap_reason, all_messages) — all_messages is the full
    pydantic-ai message log to thread into the next turn's message_history.
    When auto_select is True, populate deps.selected_patient on first
    PatientChooserEvent so follow-up clinical questions can proceed."""
    final_response = None
    cap_reason = None
    all_messages: list = []
    cur_thinking_turn = -1
    cur_text_turn = -1

    async for ev in runner.stream(question, deps=deps, message_history=message_history):
        if isinstance(ev, ThinkingDeltaEvent):
            if cur_thinking_turn != ev.turn_index:
                print(f"\n{C.GRAY}── thinking (turn {ev.turn_index}) ──{C.RESET}")
                cur_thinking_turn = ev.turn_index
            print(f"{C.GRAY}{ev.delta}{C.RESET}", end="", flush=True)

        elif isinstance(ev, ThinkingCompletedEvent):
            print(f"\n{C.DIM}  ↳ thinking done in {ev.elapsed_ms} ms{C.RESET}")

        elif isinstance(ev, AssistantTextDeltaEvent):
            if cur_text_turn != ev.turn_index:
                print(f"\n{C.MAGENTA}── streamed text (turn {ev.turn_index}) ──{C.RESET}")
                cur_text_turn = ev.turn_index
            print(ev.delta, end="", flush=True)

        elif isinstance(ev, AssistantTextCompletedEvent):
            print()  # newline after the deltas

        elif isinstance(ev, ToolCallStarted):
            print(f"\n{C.BLUE}🔧 {ev.tool_name}{C.RESET}{C.DIM}({_format_args(ev.arguments)}){C.RESET}")

        elif isinstance(ev, ToolCallCompleted):
            color = C.GREEN if ev.success else C.RED
            mark = "✓" if ev.success else "✗"
            print(f"   {color}{mark}{C.RESET} {ev.tool_name} {C.DIM}({ev.elapsed_ms} ms){C.RESET}")
            print(f"   {C.DIM}└─ {ev.result_summary}{C.RESET}")
            if ev.reliability_note:
                print(f"   {C.YELLOW}⚠  reliability: {ev.reliability_note}{C.RESET}")
            if ev.error:
                print(f"   {C.RED}error: {ev.error}{C.RESET}")
            if ev.sql_executed:
                for entry in ev.sql_executed:
                    sql = entry.get("sql", "") if isinstance(entry, dict) else str(entry)
                    params = entry.get("params") if isinstance(entry, dict) else None
                    for line in _format_sql_block(sql, params):
                        print(line)

        elif isinstance(ev, PatientChooserEvent):
            n = ev.payload.get("total_unique", 0) if isinstance(ev.payload, dict) else 0
            print(f"   {C.MAGENTA}👥 patient chooser surfaced ({n} unique matches){C.RESET}")
            if auto_select and isinstance(ev.payload, dict) and deps.selected_patient is None:
                picked = _auto_select_first(ev.payload)
                if picked is not None:
                    deps.selected_patient = picked
                    print(
                        f"   {C.MAGENTA}↳ auto-selected: {picked.display_name} (source_id={picked.source_id}){C.RESET}"
                    )

        elif isinstance(ev, CohortSampleEvent):
            sample = ev.payload.get("sample", []) if isinstance(ev.payload, dict) else []
            total = ev.payload.get("total_count", 0) if isinstance(ev.payload, dict) else 0
            print(f"   {C.MAGENTA}📊 cohort sample auto-surfaced ({len(sample)} of {total}){C.RESET}")

        elif isinstance(ev, CapReachedEvent):
            cap_reason = ev.reason
            print(f"\n{C.RED}⚠  cap reached: {ev.reason}{C.RESET}")

        elif isinstance(ev, FinalResponseEvent):
            final_response = ev.response
            all_messages = list(getattr(ev, "all_messages", []) or [])
            print(f"\n{C.BOLD}{C.GREEN}── final ──{C.RESET}")
            text = getattr(ev.response, "text", None) or "(empty)"
            print(text)
            followups = getattr(ev.response, "followups", None) or []
            for fu in followups:
                print(f"  {C.DIM}↳ followup: {fu}{C.RESET}")

    print()
    return final_response, cap_reason, all_messages


async def _replay(questions: list[str], model_override: str | None, role: str, auto_select: bool) -> int:
    """Run one or more turns through the same deps + runner, threading
    message_history between turns so follow-up questions see prior context
    (the same way the Streamlit UI does)."""
    deps = _build_deps(role)

    runner_kwargs = {}
    if model_override:
        runner_kwargs["model"] = _build_model_override(model_override)
    runner = AgenticRunner(**runner_kwargs)

    # Header
    print(f"{C.BOLD}{C.CYAN}════ agent replay ════{C.RESET}")
    print(f"  {C.DIM}role:{C.RESET}      {role}")
    if model_override:
        print(f"  {C.DIM}model:{C.RESET}     {model_override} (CLI override)")
    print(f"  {C.DIM}session:{C.RESET}   {deps.session_id}")
    print(f"  {C.DIM}turns:{C.RESET}     {len(questions)}")
    if auto_select:
        print(f"  {C.DIM}select:{C.RESET}    auto-pick first patient on chooser")
    print()

    message_history: list | None = None
    worst_exit = 0  # 0 ok, 1 no-final, 2 cap-reached
    for i, question in enumerate(questions, start=1):
        print(f"{C.BOLD}{C.CYAN}━━━ turn {i}/{len(questions)} ━━━{C.RESET}")
        print(f"  {C.DIM}question:{C.RESET}  {question}")
        if message_history is not None:
            print(f"  {C.DIM}history:{C.RESET}   {len(message_history)} prior messages threaded")
        print()

        final_response, cap_reason, all_messages = await _run_one_turn(
            runner, question, deps, message_history, auto_select
        )

        # Promote the worst exit status seen across all turns.
        if cap_reason:
            worst_exit = max(worst_exit, 2)
        elif final_response is None:
            worst_exit = max(worst_exit, 1)

        # Thread the cumulative message log into the next turn — this is
        # exactly what utils.chat_bot_helper does in the live Streamlit
        # path. Without this, follow-ups lose all coreference.
        if all_messages:
            message_history = all_messages

    return worst_exit


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Drive the production agent runner from the CLI; print every streamed event."
    )
    parser.add_argument("question", help="The first user question to send to the agent.")
    parser.add_argument(
        "--then",
        action="append",
        default=[],
        metavar="FOLLOWUP",
        help="Follow-up question(s) to send AFTER the first turn, with prior "
        "message history threaded (mirrors how the live Streamlit UI runs). "
        "Repeatable: --then 'q2' --then 'q3' chains turns 2, 3, ...",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the configured Ollama model (e.g. gpt-oss:32b, gemma4:31b). "
        "Provider-specific: only takes effect when ai_keys.provider=ollama.",
    )
    parser.add_argument(
        "--user-role",
        default="doctor",
        choices=["admin", "doctor", "nurse", "patient"],
        help="Synthetic user role for the replay (default: doctor).",
    )
    parser.add_argument(
        "--auto-select",
        action="store_true",
        help="When find_patient surfaces a chooser, auto-pick the first match (mimics "
        "the user clicking the top row in the live UI). Without this, follow-up "
        "clinical-data turns refuse with 'no patient slot.'",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colors (auto-disabled when piped).",
    )
    args = parser.parse_args()

    if args.no_color or not sys.stdout.isatty():
        _disable_color()

    questions = [args.question, *args.then]
    return asyncio.run(_replay(questions, args.model, args.user_role, args.auto_select))


if __name__ == "__main__":
    sys.exit(main())
