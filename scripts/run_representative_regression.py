"""Live-model regression runner for the §13 representative-question suite.

Phase 2 acceptance gate: ≥80% pass rate. Below 80% blocks merge.

The LLM path is live (Ollama by default per secrets.toml). The data
layer is the same in-memory SQLite synthetic mirror that the unit tests
use — see `tests/agent/redshift_synthetic.sql`. The suite hardcodes
`source_id='src-john-1962'`, the diabetic patient in that fixture, so
running against prod Redshift would surface no records and fail the
gate for the wrong reason.

Usage:
    uv run python scripts/run_representative_regression.py
    uv run python scripts/run_representative_regression.py --suite path/to.yaml
    uv run python scripts/run_representative_regression.py --ids Q4 Q6
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import date, datetime
from pathlib import Path

import yaml
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

# sys.path shim — same pattern as scripts/seed_agent_rag.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.db.analytics_adapter import AnalyticsDbAdapter
from agent.deps import AgentDeps, SelectedPatient
from agent.observability import configure_observability
from agent.runner import AgenticRunner
from agent.state import (
    FinalResponseEvent,
    ToolCallStarted,
    ToolCallCompleted,
)
from unittest.mock import MagicMock


_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_SUITE = _REPO_ROOT / "tests/agent/regression/representative_questions.yaml"
_SYNTHETIC_SQL = _REPO_ROOT / "tests/agent/redshift_synthetic.sql"
_DEFAULT_PATIENT = SelectedPatient(
    source_id="src-john-1962",
    display_name="John Smith",
    dob=date(1962, 5, 1),
    selected_at=datetime.now(),
    selection_origin="user_click",
)


def _build_synthetic_analytics_db() -> AnalyticsDbAdapter:
    """In-memory SQLite mirror of the §7.12 view whitelist.

    StaticPool + check_same_thread=False are required because pydantic-ai
    runs sync tools in run_in_executor across threads. Without them, the
    `:memory:` database created on the main thread is invisible to the
    worker thread that actually invokes the SQL templates.
    """
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    sql = _SYNTHETIC_SQL.read_text()
    with engine.begin() as conn:
        for stmt in sql.split(";"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(text(stmt))
    return AnalyticsDbAdapter(engine=engine, dialect="sqlite")


def _build_live_deps() -> AgentDeps:
    """Build deps for a regression run.

    LLM is live (Ollama / Anthropic / Bedrock per secrets.toml).
    Analytics DB is the synthetic in-memory SQLite mirror — see module
    docstring for why. RAG reads from the local Chroma store, which
    must already be seeded via `scripts/seed_agent_rag.py`.
    """
    import streamlit as st  # noqa: F401 — imported for secrets access

    from agent.rag.chroma_adapter import ChromaRagAdapter
    import chromadb
    from orm.models import RoleTypeEnum, SessionLocal

    chroma_path = st.secrets.get("rag_model", {}).get("chroma_path", "./chromadb")
    rag = ChromaRagAdapter(chromadb.PersistentClient(path=chroma_path))

    return AgentDeps(
        user_id=0,
        user_role=RoleTypeEnum.DOCTOR,
        session_id=f"regression-{datetime.now().isoformat(timespec='seconds')}",
        selected_patient=_DEFAULT_PATIENT,
        last_dataframe=None,
        last_sql=None,
        last_query_meta=None,
        analytics_db=_build_synthetic_analytics_db(),
        rag=rag,
        sqlite_session=SessionLocal(),
        audit_logger=MagicMock(),  # regression runs don't write to audit
    )


def _grade(question: dict, tool_calls: list[dict], final_text: str) -> dict:
    expected_tools = set(question["expected_tools"])
    actual_tool_names = {tc["tool_name"] for tc in tool_calls}
    tools_pass = expected_tools.issubset(actual_tool_names)

    expected_da = question["expected_data_availability"]
    principal_tools = {"get_patient_clinical_data", "list_patient_documents"}
    principal_calls = [tc for tc in tool_calls if tc["tool_name"] in principal_tools]
    if principal_calls:
        actual_das = {
            (tc.get("result_summary") or "").split("data_availability=")[-1].split(";")[0].strip()
            for tc in principal_calls
        }
        da_pass = expected_da in actual_das
    else:
        # Q10 expects domain_not_available without a clinical-data call —
        # that's a pass by definition.
        da_pass = expected_da == "domain_not_available"

    final_lower = (final_text or "").lower()
    must_not = question.get("must_not_contain", []) or []
    must_not_pass = all(p.lower() not in final_lower for p in must_not)

    must_any = question.get("must_contain_any", []) or []
    must_any_pass = (not must_any) or any(p.lower() in final_lower for p in must_any)

    overall_pass = tools_pass and da_pass and must_not_pass and must_any_pass
    return {
        "tools_pass": tools_pass,
        "data_availability_pass": da_pass,
        "must_not_contain_pass": must_not_pass,
        "must_contain_any_pass": must_any_pass,
        "pass": overall_pass,
        "expected_tools": sorted(expected_tools),
        "actual_tools": sorted(actual_tool_names),
    }


async def _run_one(runner: AgenticRunner, deps: AgentDeps, prompt: str) -> tuple[list[dict], str]:
    # selected_patient is on deps; agent.instructions.selection_instructions
    # surfaces it to the model on every run, so no prompt-prefix is needed.
    tool_calls: list[dict] = []
    final_text = ""
    async for evt in runner.stream(prompt, deps=deps):
        if isinstance(evt, ToolCallStarted):
            tool_calls.append({"tool_name": evt.tool_name, "result_summary": ""})
        elif isinstance(evt, ToolCallCompleted):
            for tc in reversed(tool_calls):
                if tc["tool_name"] == evt.tool_name and not tc["result_summary"]:
                    tc["result_summary"] = evt.result_summary
                    break
        elif isinstance(evt, FinalResponseEvent):
            final_text = evt.response.text
    return tool_calls, final_text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, default=_DEFAULT_SUITE)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Minimum pass rate (0–1) required for exit code 0",
    )
    parser.add_argument("--ids", nargs="*", default=None, help="Subset of question IDs to run")
    args = parser.parse_args()

    suite = yaml.safe_load(args.suite.read_text())
    questions = suite["questions"]
    if args.ids:
        questions = [q for q in questions if q["id"] in args.ids]

    configure_observability()
    deps = _build_live_deps()
    runner = AgenticRunner()

    results: list[dict] = []
    loop = asyncio.new_event_loop()
    try:
        for q in questions:
            print(f"\n=== {q['id']}: {q['prompt']}\n")
            try:
                tool_calls, final_text = loop.run_until_complete(_run_one(runner, deps, q["prompt"]))
            except Exception as exc:
                # Walk the __cause__ chain — pydantic-ai wraps the underlying
                # ValidationError inside UnexpectedModelBehavior, so the chain
                # is where the actually useful "date_range expected object got
                # string" message lives.
                detail = f"{type(exc).__name__}: {exc}"
                cause = exc.__cause__ or exc.__context__
                while cause is not None:
                    detail += f"\n      caused by {type(cause).__name__}: {cause}"
                    cause = cause.__cause__ or cause.__context__
                print(f"  [✗] crashed: {detail}")
                results.append(
                    {
                        "id": q["id"],
                        "tools_pass": False,
                        "data_availability_pass": False,
                        "must_not_contain_pass": False,
                        "must_contain_any_pass": False,
                        "pass": False,
                        "final_text": "",
                        "expected_tools": sorted(set(q.get("expected_tools", []))),
                        "actual_tools": [],
                        "crashed": str(exc),
                    }
                )
                continue
            grade = _grade(q, tool_calls, final_text)
            results.append({"id": q["id"], **grade, "final_text": final_text})
            flag = "✓" if grade["pass"] else "✗"
            print(
                f"  [{flag}] tools={grade['tools_pass']} da={grade['data_availability_pass']} "
                f"must_not={grade['must_not_contain_pass']} must_any={grade['must_contain_any_pass']}"
            )
    finally:
        loop.close()

    total = len(results)
    passed = sum(1 for r in results if r["pass"])
    rate = passed / total if total else 0.0
    print(f"\nPass rate: {passed}/{total} = {rate:.0%}  (threshold: {args.threshold:.0%})")
    return 0 if rate >= args.threshold else 1


if __name__ == "__main__":
    raise SystemExit(main())
