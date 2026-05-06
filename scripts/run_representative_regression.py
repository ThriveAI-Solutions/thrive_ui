"""Live-model regression runner for the §13 representative-question suite.

Phase 2 acceptance gate: ≥80% pass rate. Below 80% blocks merge.

Usage:
    uv run python scripts/run_representative_regression.py
    uv run python scripts/run_representative_regression.py --suite path/to.yaml
    RUN_CLOUD_EVAL=1 uv run python scripts/run_representative_regression.py

The cloud path is an opt-in (Bedrock / Anthropic) controlled by
secrets.toml + RUN_CLOUD_EVAL=1.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import date, datetime
from pathlib import Path

import yaml

# sys.path shim — same pattern as scripts/seed_agent_rag.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.deps import AgentDeps, SelectedPatient
from agent.observability import configure_observability
from agent.runner import AgenticRunner
from agent.state import (
    FinalResponseEvent,
    ToolCallStarted,
    ToolCallCompleted,
)
from unittest.mock import MagicMock


_DEFAULT_SUITE = Path(__file__).resolve().parent.parent / "tests/agent/regression/representative_questions.yaml"
_DEFAULT_PATIENT = SelectedPatient(
    source_id="src-john-1962",
    display_name="John Smith",
    dob=date(1962, 5, 1),
    selected_at=datetime.now(),
    selection_origin="user_click",
)


def _build_live_deps() -> AgentDeps:
    """Build deps for a live-model run.

    Reads the same secrets.toml as the Streamlit app (analytics_db,
    ai_keys, rag_model). Patient is pre-selected so the agent doesn't
    have to disambiguate.
    """
    import streamlit as st  # noqa: F401  — imported for secrets access

    from agent.db.analytics_adapter import AnalyticsDbAdapter
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
        analytics_db=AnalyticsDbAdapter.from_streamlit_secrets(),
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
    for q in questions:
        print(f"\n=== {q['id']}: {q['prompt']}\n")
        tool_calls, final_text = asyncio.run(_run_one(runner, deps, q["prompt"]))
        grade = _grade(q, tool_calls, final_text)
        results.append({"id": q["id"], **grade, "final_text": final_text})
        flag = "✓" if grade["pass"] else "✗"
        print(
            f"  [{flag}] tools={grade['tools_pass']} da={grade['data_availability_pass']} "
            f"must_not={grade['must_not_contain_pass']} must_any={grade['must_contain_any_pass']}"
        )

    total = len(results)
    passed = sum(1 for r in results if r["pass"])
    rate = passed / total if total else 0.0
    print(f"\nPass rate: {passed}/{total} = {rate:.0%}  (threshold: {args.threshold:.0%})")
    return 0 if rate >= args.threshold else 1


if __name__ == "__main__":
    raise SystemExit(main())
