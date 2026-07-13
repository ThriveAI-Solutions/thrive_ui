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

## Authenticated Evaluation Workspace (in-app)

The CLI above is a developer tool. The **authenticated evaluation workspace**
is the product surface: admins launch, monitor, and review agent evaluations
inside the app, and end users leave thumbs feedback on completed agentic
answers. Design + plan:
`docs/superpowers/specs/2026-07-10-authenticated-evaluation-workspace-design.md`
and `docs/superpowers/plans/2026-07-10-authenticated-evaluation-workspace.md`.

### Agentic feedback vs Vanna training

- Thumbs up/down on a **completed agentic answer** (`views/agent_feedback.py`)
  records *product feedback only* — it is owned by the user who left it,
  auditable (`thrive_agent_run_feedback` + `_event`), and **never invokes Vanna
  training**. Only the originating user can change or clear their feedback.
- The legacy Vanna thumbs/training flow (`views/admin_feedback.py`,
  `utils.vanna_calls.write_to_file_and_training`) is unchanged and independent.

### Cases: feedback snapshots and curated

- A thumbs-down interaction is snapshotted into an **immutable
  `EvaluationCase`** (`evals/cases.py::snapshot_feedback_case`) capturing the
  exact question, patient source_id, history, tool evidence, original answer,
  and the reviewer's concern. Later feedback edits or log retention cannot
  change what a re-run replays. If the originating run had
  `[agent_logging].mode = "disabled"`, exact replay is unavailable and
  snapshotting raises `SnapshotUnavailable`.
- Curated cases are the parameterized YAML conversations (same normalization).
  Promotion (`promote_feedback_case`) generalizes a reviewed feedback case into
  a **draft curated case** — it persists an app draft and **never edits
  `evals/questions.yaml`**.

### Synchronous vs asynchronous selection

- Exactly **one feedback case** runs **synchronously** in the request.
- **Every other selection** — two or more feedback cases, any curated
  selection, or the full curated suite — runs **asynchronously** on the worker.
- Only **one active asynchronous run per deployment** is permitted; it is
  enforced transactionally in SQLite (`BEGIN IMMEDIATE` in the claim), so extra
  queued runs wait rather than run concurrently.

### Durable statuses, worker, and recovery

- Runs (`thrive_evaluation_run`) and per-case results
  (`thrive_evaluation_case_result`) are durable, so browser reruns and process
  restarts never lose progress. Run statuses: `queued`, `running`,
  `completed`, `completed_with_errors`, `failed`, `cancelled`.
- A **process-local daemon worker** (`evals/worker.py`) starts once after DB
  bootstrap (`app.py`), polls every 5 s, claims one queued async run at a time,
  heartbeats before/after each case, and commits each case result
  independently. If a worker dies, a run whose heartbeat is older than the
  timeout (120 s) is requeued and resumes only its unfinished cases. The
  database claim stays authoritative even if more than one app process runs.
  Cancellation stops after the current case and marks remaining cases
  `cancelled`.

### Verdicts, logging mode, and retention

- The local-LLM judge is **triage only** (`looks_correct` / `looks_wrong` /
  `unsure`). The authoritative verdict is an admin's `correct`, `incorrect`, or
  `cant_tell`, with full change history in `thrive_evaluation_review_event`.
- `[agent_logging].mode` fidelity is honored: `scrubbed` snapshots carry only
  PHI-safe summaries (no full result rows); `disabled` cannot be replayed.
- Retention: expired case payloads and result rows are purged (case status
  `expired`); run identity/status/counts/timestamps and review audit rows are
  always retained.
- **Notifications are PHI-free** — only run id, status, counts, and timestamps
  (`Evaluation <run_id> completed: <completed>/<total> cases.`). Production
  reports remain **authenticated application pages** — the standalone JSON/HTML
  the CLI writes is a developer capability, never the production report store.

### Worker configuration

Add to `.streamlit/secrets.toml` (defaults shown; the worker runs with these
even if the section is absent):

```toml
[agent_evaluations]
worker_enabled = true
poll_interval_s = 5
heartbeat_timeout_s = 120
```
