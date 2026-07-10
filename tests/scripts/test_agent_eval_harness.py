"""CLI contract tests for scripts/agent_eval_harness.py: it stays a thin
adapter over evals.cases + evals.executor, preserving --dry-run, JSON
output shape, and the other existing switches."""

import json
from pathlib import Path

import pytest

from agent.state import AgentResponse, FinalResponseEvent
from scripts import agent_eval_harness as harness

_ROSTER_ONE_PATIENT = """
defaults:
  date_start: "2024-01-01"
  date_end: "2026-06-11"
  vaccine: "MMR"
  disease: "diabetes"
patients:
  - source_id: "src-a"
    label: "patient A"
"""


@pytest.fixture
def roster_path(tmp_path: Path) -> Path:
    p = tmp_path / "roster.yaml"
    p.write_text(_ROSTER_ONE_PATIENT)
    return p


class _FakeAdapter:
    schema_prefix = ""

    def __init__(self):
        self.popped = 0

    def fetch_all(self, sql, params=None):
        return [{"display_name": "Jane Doe", "dob": "1980-01-01"}]

    def pop_sql_log(self):
        self.popped += 1
        return []


class _FakeAdapterFactory:
    instances: list = []

    @classmethod
    def from_streamlit_secrets(cls):
        adapter = _FakeAdapter()
        cls.instances.append(adapter)
        return adapter


class _FakeRunner:
    async def stream(self, prompt, deps=None, message_history=None):
        yield FinalResponseEvent(response=AgentResponse(text=f"Answer to: {prompt}"), all_messages=[], usage=None)


def test_dry_run_prints_resolved_matrix_and_exits_zero(monkeypatch, roster_path, capsys):
    monkeypatch.setattr(
        "sys.argv",
        ["agent_eval_harness.py", "--roster", str(roster_path), "--only", "Q1", "--dry-run"],
    )

    exit_code = harness.main()

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "1 conversations" in out
    assert "src-a" in out


def test_missing_roster_returns_error_exit_code(monkeypatch, tmp_path, capsys):
    missing = tmp_path / "does-not-exist.yaml"
    monkeypatch.setattr("sys.argv", ["agent_eval_harness.py", "--roster", str(missing)])

    exit_code = harness.main()

    assert exit_code == 2
    assert "not found" in capsys.readouterr().out


def test_live_run_writes_incremental_json_in_legacy_shape(monkeypatch, roster_path, tmp_path):
    monkeypatch.setattr(harness, "AnalyticsDbAdapter", _FakeAdapterFactory)
    monkeypatch.setattr(harness, "AgenticRunner", _FakeRunner)
    monkeypatch.setattr(harness, "configure_observability", lambda: None)
    monkeypatch.setattr(harness, "_build_rag", lambda: None)

    out_path = tmp_path / "eval-test.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "agent_eval_harness.py",
            "--roster",
            str(roster_path),
            "--only",
            "Q1",
            "--skip-judge",
            "--out",
            str(out_path),
        ],
    )

    exit_code = harness.main()

    assert exit_code == 0
    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert data["run_id"] == "eval-test"
    (convo,) = data["conversations"]
    assert convo["conversation_id"] == "Q1__src-a"
    assert convo["question_id"] == "Q1"
    assert convo["status"] == "ok"
    assert convo["patient"]["source_id"] == "src-a"
    assert convo["patient"]["display_name"] == "Jane Doe"
    assert convo["turns"][0]["answer"].startswith("Answer to:")
    assert convo["turns"][0]["judge"] is None
