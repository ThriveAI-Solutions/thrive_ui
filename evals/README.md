# Agentic Eval Harness

Runs the 10 representative analyst questions (sub-questions as follow-up
turns) through the `agent/` pipeline against the prod warehouse, then
generates a self-contained HTML report where reviewers mark each answer
Correct / Incorrect / Can't tell. Design:
`docs/superpowers/specs/2026-06-11-agentic-eval-harness-design.md`.

## Quick start

```bash
# 1. Seed the roster (real source_ids — file is gitignored, results carry PHI)
cp evals/roster.example.yaml evals/roster.yaml
uv run python scripts/agent_eval_harness.py --suggest-patients   # optional candidates

# 2. Preview the matrix without touching the LLM or warehouse
uv run python scripts/agent_eval_harness.py --dry-run

# 3. Cheap smoke, then the full run (expect ~1h for 3 patients × 10 questions)
uv run python scripts/agent_eval_harness.py --only Q4 --skip-judge
uv run python scripts/agent_eval_harness.py

# 4. Generate + open the report
uv run python scripts/generate_eval_report.py evals/results/<run_id>.json
open evals/results/<run_id>.html
```

Marking persists in the browser's localStorage (keyed by run id) and via
the report's Export/Import verdict buttons (JSON/CSV) for pass-the-file
review. The judge chip is triage from the local LLM — humans make every
final call.

Latency attribution: each turn splits wall clock into LLM / our code /
warehouse using per-query `db_elapsed_ms` recorded by the analytics adapter.
