# Authenticated Evaluation Workspace — Swarm Constitution

## Done-right

1. Completed agentic answer groups expose exactly one owned thumbs-up/thumbs-down control set; intermediate reasoning, tool, SQL, chart, table, chooser, and error-only artifacts never do.
2. Agentic feedback is idempotent, auditable, linked to the completed AgentRun/final message, and never invokes Vanna training.
3. Admins can browse normalized feedback and curated cases, launch one feedback replay synchronously, and launch feedback batches or curated runs asynchronously.
4. Evaluation runs, immutable case snapshots, independent case results, heartbeat/recovery state, cancellation, final admin verdicts, promotion drafts, and PHI-free notifications survive browser reruns and process restarts.
5. Every evaluation read/mutation rechecks database-backed authorization; only the feedback owner can change feedback and only admins can launch or view evaluations.
6. LLM verdicts remain triage metadata; only `correct`, `incorrect`, or `cant_tell` admin verdicts are authoritative.
7. Logging modes and retention are enforced: scrubbed evidence remains scrubbed, disabled logging cannot claim exact replay, and expired PHI payloads are removed while audit metadata remains.
8. Existing curated CLI dry-run/live/JSON/optional HTML behavior and existing Vanna thumbs/training behavior remain compatible.
9. `uv run ruff check` and `uv run pytest` pass on the integrated epic branch.
10. The final PR is opened against `main` and excludes `.ma/`, `.ma-worktrees/`, credentials, generated evaluation outputs, and local databases.

## Repo rules

- Repository: `/Users/kyleroot/Code/thrive/thrive_ui`; default branch: `main`; epic branch: `ma/auth-eval-workspace`.
- Read and follow `CLAUDE.md` before editing.
- Use `uv`; focused tests during work, then `uv run ruff check` and `uv run pytest` before final review.
- Database changes use SQLAlchemy models plus reviewed Alembic revisions; verify upgrade and downgrade on temporary SQLite databases.
- Keep SQLite app state separate from the read-only analytics warehouse.
- Match neighboring Streamlit and pytest conventions; no drive-by refactors or formatting.
- Commit only on the assigned task branch with a conventional commit message. Workers never merge or push.
- Do not expose PHI in logs, process titles, notifications, commit messages, card comments, or test fixtures.
- Do not edit `.env`, credentials, deployment secrets, local databases, or repository YAML from production request paths.
- Do not commit directly to `main` or `master`.

## Evidence rule

Every completion claim must cite an executed command, exit code, observed output tail, touched files, and commit SHA. Checkers rerun the contract commands independently. Mock-only evidence cannot establish migration, concurrency, persistence, or CLI compatibility where the contract calls for a real temporary SQLite DB or real dry run.

## Protected invariants

- Existing Vanna feedback/training implementation and tests remain behaviorally unchanged.
- `evals/questions.yaml` is read-only to application/runtime code; promotion persists an app draft only.
- Analytics adapters remain read-only and patient resolution fails closed; no fallback to a different patient.
- Standalone PHI-bearing HTML is never linked from production UI and is never the production report store.
- Agent logging mode `disabled` does not persist question, patient, answer, history, tool payloads, usage, errors, or stack traces in AgentRun logging tables; only the minimal run identity needed to own feedback may persist.
- Notifications contain only run ID, status/counts, timestamps, and action identity.
- One active asynchronous evaluation run per deployment is enforced transactionally in SQLite.
- The asynchronous worker is a process-level singleton and starts only after successful database bootstrap.
- Public route/service reads reject non-admin users even if called without the Streamlit page guard.
- Final branch diff excludes `.ma/` and `.ma-worktrees/`; `.ma/constitution.md` is run-only bookkeeping.

## File ownership

Each contract owns only its listed boundaries. Shared registration files (`orm/models.py`, `views/admin.py`, `app.py`) may be touched only by the single task that owns them. New modules are used to keep later tasks from reopening earlier implementation files. A worker encountering a required out-of-bound edit must block rather than edit it.
