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

## Local sample DB run (no warehouse, no PHI)

To exercise the agent end-to-end without the prod warehouse — useful for
verifying tool/agent changes — point `[analytics_db]` at the committed
synthetic sample instead of the live warehouse:

```bash
# 1. Build a sqlite copy of the sample warehouse from the committed dump
uv run python scripts/load_sample_db.py --target sqlite --path ./sample_warehouse.db

# 2. In .streamlit/secrets.toml point the agent's warehouse at it:
#   [analytics_db]
#   dialect = "sqlite"
#   url     = "sqlite:///./sample_warehouse.db"

# 3. Roster source_ids must be SAMPLE source_ids. Find candidates in the sqlite, e.g.:
#   sqlite3 sample_warehouse.db \
#     "SELECT source_id, med_name FROM federated_meds_v WHERE lower(med_name) LIKE '%doxycycline%' LIMIT 5;"
#   then put one in evals/roster.yaml.

# 4. Run a focused single-turn smoke (evals/q9_meds_smoke.yaml is one such question):
uv run python scripts/agent_eval_harness.py --questions evals/q9_meds_smoke.yaml \
    --only Q9 --skip-judge --limit-patients 1
```

Notes:
- **Model matters.** Use a capable model (`gpt-oss:*`, `qwen3.6:27b`). Small
  models like `gemma4` emit invalid strict-tool inputs and fail with
  `UnexpectedModelBehavior: ... exceeded max retries`. Ensure Ollama is running
  and the model is pulled.
- `sample_warehouse.db` is a generated artifact (gitignored) — rebuild it from
  `data/sample/thrive_sample.sql.zst` anytime with the step-1 command.
- The sample data has empty `status`/`date_stopped` and synthetic (often very
  old) `date_prescribed` values — widen the roster `date_start` if a question
  filters by date.
