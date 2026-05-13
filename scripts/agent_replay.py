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

from agent.audit import AuditLogger  # noqa: E402
from agent.db.analytics_adapter import AnalyticsDbAdapter  # noqa: E402
from agent.deps import AgentDeps  # noqa: E402
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


def _build_deps(role: str) -> AgentDeps:
    """Stub the session-state bits build_agent_deps reads from Streamlit;
    everything else (Redshift adapter, ChromaDB) uses the same factories
    the live app does so behavior matches."""
    import streamlit as st

    analytics_db = AnalyticsDbAdapter.from_streamlit_secrets()
    chroma_path = st.secrets.get("rag_model", {}).get("chroma_path", "./chromadb")
    rag = ChromaRagAdapter(chromadb.PersistentClient(path=chroma_path))

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


async def _replay(question: str, model_override: str | None, role: str) -> int:
    deps = _build_deps(role)

    runner_kwargs = {}
    if model_override:
        runner_kwargs["model"] = _build_model_override(model_override)
    runner = AgenticRunner(**runner_kwargs)

    # Header
    print(f"{C.BOLD}{C.CYAN}════ agent replay ════{C.RESET}")
    print(f"  {C.DIM}question:{C.RESET}  {question}")
    print(f"  {C.DIM}role:{C.RESET}      {role}")
    if model_override:
        print(f"  {C.DIM}model:{C.RESET}     {model_override} (CLI override)")
    print(f"  {C.DIM}session:{C.RESET}   {deps.session_id}")
    print()

    final_response = None
    cap_reason = None
    cur_thinking_turn = -1
    cur_text_turn = -1

    async for ev in runner.stream(question, deps=deps):
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

        elif isinstance(ev, CohortSampleEvent):
            sample = ev.payload.get("sample", []) if isinstance(ev.payload, dict) else []
            total = ev.payload.get("total_count", 0) if isinstance(ev.payload, dict) else 0
            print(f"   {C.MAGENTA}📊 cohort sample auto-surfaced ({len(sample)} of {total}){C.RESET}")

        elif isinstance(ev, CapReachedEvent):
            cap_reason = ev.reason
            print(f"\n{C.RED}⚠  cap reached: {ev.reason}{C.RESET}")

        elif isinstance(ev, FinalResponseEvent):
            final_response = ev.response
            print(f"\n{C.BOLD}{C.GREEN}── final ──{C.RESET}")
            text = getattr(ev.response, "text", None) or "(empty)"
            print(text)
            followups = getattr(ev.response, "followups", None) or []
            for fu in followups:
                print(f"  {C.DIM}↳ followup: {fu}{C.RESET}")

    print()
    if cap_reason:
        return 2
    return 0 if final_response is not None else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Drive the production agent runner from the CLI; print every streamed event."
    )
    parser.add_argument("question", help="The user question to send to the agent.")
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
        "--no-color",
        action="store_true",
        help="Disable ANSI colors (auto-disabled when piped).",
    )
    args = parser.parse_args()

    if args.no_color or not sys.stdout.isatty():
        _disable_color()

    return asyncio.run(_replay(args.question, args.model, args.user_role))


if __name__ == "__main__":
    sys.exit(main())
