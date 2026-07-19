# Repository Working Agreement

These instructions apply to the entire repository. They are the operating
contract for agents and contributors working in `thrive_ui`.

## Mission and priorities

HEALTHeINTELLIGENCE is a healthcare data application. Optimize for correctness,
traceability, least privilege, and safe handling of sensitive data before speed
or cleverness. A plausible answer with weak provenance is worse than an explicit
"not enough information" result.

When guidance conflicts, follow this order:

1. The user's explicit request and documented business decisions.
2. Security, privacy, consent, and compliance requirements.
3. This file and narrower `AGENTS.md` files, if any are added later.
4. The repository's executable contracts: tests, migrations, configuration, and
   implementation.
5. README prose and historical documentation.

Do not silently resolve contradictions. Record the conflict in the relevant
issue or pull request and ask for the missing decision when it changes behavior.

## Repository map

| Area | Purpose |
| --- | --- |
| `app.py`, `views/` | Streamlit entry point, chat, admin, audit, and evaluation UI |
| `agent/` | Agentic orchestration, model selection, tools, RAG, and analytics access |
| `utils/` | Authentication, legacy Vanna path, renderers, configuration, and shared utilities |
| `utils/llm_registry/` | Provider registration and runtime LLM construction |
| `orm/` | SQLAlchemy models and application/audit persistence |
| `alembic/` | Versioned application-database migrations |
| `evals/` | Curated evaluation cases, run persistence, review, and CLI/UI support |
| `tests/` | Unit, integration, view, ORM, sample-database, and evaluation tests |
| `scripts/` | Deployment, migration, sample-data, and maintenance helpers |
| `llmdocs/`, `docs/superpowers/specs/` | Database notes and detailed design records |
| `.github/workflows/` | GitHub Actions; merging to `dev` triggers the dev deployment workflow |

Prefer an existing module and its tests over introducing a parallel abstraction.
Before changing a cross-cutting behavior, search both the agentic and legacy
paths; they coexist and may require deliberate parity.

### Canonical references

Read the relevant detailed reference before changing the associated subsystem:

- `CLAUDE.md`: current configuration map, module-level extension points, and
  repository-specific gotchas. This file remains the higher-level operating
  contract if the two disagree.
- `llmdocs/database.md`: application SQLite, analytics PostgreSQL/Redshift, and
  vector-store boundaries.
- `docs/superpowers/specs/2026-05-06-agentic-replatform-design.md`: canonical
  design for agent tools, caps, event streaming, and run logging.
- `docs/superpowers/specs/2026-05-13-sample-database-design.md` and
  `data/sample/README.md`: synthetic sample-database setup and intended use.
- `docs/superpowers/specs/2026-06-26-simplify-medications-retrieval-design.md`:
  medication retrieval rationale and code-granularity constraints.

Treat dated specs as decision history, not automatic proof that every proposed
detail is still implemented. Verify the current code and tests.

## Local setup and authoritative commands

The project requires Python 3.13 and uses `uv` for the locked environment.

```bash
uv sync
uv run pre-commit install
cp .streamlit/secrets_example.toml .streamlit/secrets.toml  # local only
docker compose up -d
uv run python scripts/load_sample_db.py  # synthetic data only; optional
uv run streamlit run app.py
```

Common validation commands:

```bash
uv run ruff check <changed Python paths>
uv run ruff format --check <changed Python paths>
uv run pytest -q <focused test paths>
uv run pytest -q
uv run pytest -m "not milvus and not sample_db"
uv run pre-commit run --all-files
```

Useful marked suites:

- `milvus`: requires Milvus Lite and must use an isolated temporary database.
- `sample_db`: requires the sample dataset in a running PostgreSQL instance.
- `integration` and `slow`: run when the change crosses process, storage, or
  external-service boundaries.

Use `uv run` so local results use the repository's locked tools. Do not install
ad-hoc packages globally or mutate `uv.lock` unless dependency changes are part
of the issue.

## GitHub delivery workflow

Unless the user explicitly asks for a different workflow, substantive changes
must be delivered through GitHub:

1. Identify the existing issue before coding, or create a narrowly scoped issue
   when the work is not tracked.
2. Check the GitHub Project and issue labels before assuming ownership. An issue
   assigned to Thrive may still be a HEALTHeLINK responsibility. Treat
   `thrive`, `healthelink`, `joint`, and `external-dependency` labels as the
   ownership source of truth, and call out contradictions.
3. Move the Project item to `In Progress` when implementation begins. Use
   `Blocked` only for a real dependency or missing decision, not ordinary work.
4. Work on a dedicated branch. Codex-created branches use the `codex/` prefix.
5. Keep unrelated working-tree changes out of the branch, commit, and pull
   request. Stage explicit paths in a mixed worktree; do not use `git add -A`.
6. Run focused tests for the changed behavior and a proportionate broader suite.
   Record the exact commands and results in the pull request.
7. Open a pull request that explains the problem, root cause, implementation,
   impact, risks, and validation. Use `Closes #N` only when the merged change
   fully satisfies that issue; otherwise use `Refs #N` and state what remains.
8. Merge only after required checks pass and review requirements are satisfied.
   Do not weaken tests or branch protections to make a change mergeable.
9. After merge, reconcile GitHub: comment with evidence, close completed issues,
   and update Project status (`Todo`, `In Progress`, `Blocked`, `In Review`, or
   `Done`). Never close an external dependency merely because Thrive's portion
   is complete.

## Ownership and decision boundaries

The following areas require an approved business or partner decision. Do not
invent policy in code, fixtures, documentation, or issue comments:

- consent enforcement, exemptions, blanks, break-glass behavior, and ownership;
- patient identity resolution and post-Fusion disambiguation rules;
- Public Health Okta groups, role mapping, and permitted capabilities;
- SHIN-NY minimum audit dataset, retention, and compliance acceptance;
- validation thresholds, authoritative patient rosters, and known answers.

When Thrive's implementation is complete but HeL validation remains, leave the
issue open, attach implementation evidence, label the dependency accurately,
and move it to `In Review` or `Blocked` as appropriate.

## Implementation guardrails

### General Python changes

- Keep diffs narrow and preserve public behavior unless the issue explicitly
  changes it. Avoid opportunistic renames and broad formatting churn.
- Add or update tests with behavior changes. A regression fix should normally
  include a test that fails before the fix.
- Prefer explicit types and small, testable functions at service boundaries.
- Do not catch broad exceptions without logging actionable context. Never turn
  an authorization, query, or provider failure into a misleading "no data"
  result.
- Preserve backward compatibility for persisted rows, settings, and serialized
  evaluation data unless a migration and rollout plan says otherwise.

### Streamlit UI and session state

- Treat every interaction as a full script rerun. Initialize session-state keys
  deterministically before reading them.
- Keep rendering separate from side effects. Authentication, writes, model calls,
  and long-running work must not repeat merely because a widget rerendered.
- Use forms or explicit actions for mutations. Disable repeated submissions while
  work is active and surface recoverable errors to the user.
- Cache only data that is safe to share at the cache's scope. Never cache
  user-specific clinical results globally.
- Maintain both visible behavior and audit behavior when changing chat, admin,
  authentication, or evaluation flows.
- Preserve established cross-rerun keys unless the change includes a migration
  of every reader and writer: `my_question`, `messages`, `last_run_sql_error`,
  `last_failed_sql`, `pending_sql_error`, `streamed_summary`, and `_vn_instance`.

Common extension paths:

- A new persisted user setting normally requires a model column, Alembic
  revision, load/save support in `orm/functions.py`, and the relevant UI control.
- A new message type normally requires `utils.enums.MessageType`, a renderer,
  `MESSAGE_RENDERERS` registration, persistence/serialization coverage, and a
  rerun-safe rendering test.

### Agent and LLM behavior

- Honor the runtime provider/model selection; do not bypass the LLM registry with
  a hard-coded provider or model.
- Keep provider-specific construction in `utils/llm_registry/providers/` and
  provider-neutral orchestration in the agent/runtime layers.
- Mock network/provider calls in unit tests. Tests must not require real API keys,
  spend tokens, or transmit prompts or clinical context externally.
- Tool outputs must distinguish success, empty results, unavailable data,
  authorization failures, and execution errors. Preserve provenance needed for
  the final answer and audit trail.
- Changes to prompts, tools, fallback behavior, or result synthesis require tests
  for the affected agentic path and, where applicable, the legacy Vanna path.
- Preserve configured hard caps such as `max_tool_calls` and
  `max_wall_clock_s`. A retry or fallback must not create an unbounded second
  execution path.

### SQL and clinical data access

- Use the analytics adapter and established query modules. Do not open a new
  direct warehouse connection from a view or prompt helper.
- Parameterize values and validate identifiers. Never interpolate patient input,
  free text, table names, or sort expressions into SQL without an approved,
  allow-listed construction.
- Account explicitly for PostgreSQL/Redshift dialect and type differences. Do
  not "fix" tests by weakening casts or bypassing the intended adapter.
- Enforce role, patient scope, and consent at the authoritative data-access
  boundary once the governing rules are approved. UI filtering alone is not an
  access control.
- Bound result sizes and avoid logging raw clinical rows. A query that touches a
  patient must preserve the corresponding patient-access audit event.

### Clinical-domain contracts and sharp edges

- Medication retrieval intentionally returns the patient's medication list and
  filters by `med_name`; do not reintroduce an RxNorm drug-class filter. RxNorm
  ingredient/clinical-drug granularity does not match that use case.
- Warehouse medication numeric fields may contain `''`. Preserve the
  `BlankableInt` behavior for `drug_supply_days` and `number_of_refills`; a strict
  integer model turns valid rows into misleading empty results and retry loops.
- `federated_allergies_v` uses the quoted `"date"` column plus `created_date`; it
  does not provide `onset_date` or `status_datetime`. Keep the query aliases and
  sample schema/catalog tests synchronized.
- Vocabulary tables (`vocab_*`) live in the application database, not git.
  Missing data should surface `VocabNotLoadedError`; load the synthetic/local dump
  with `uv run python scripts/import_vocab_dump.py data/vocab/` rather than
  hard-coding vocabulary JSON back into the application.
- When a production-schema discrepancy is confirmed, update the analytics query,
  synthetic schema/fixtures, and schema-contract tests together. Never encode a
  guessed production column merely to satisfy an LLM-generated query.

### Persistence and migrations

- Treat `orm/models.py` and Alembic revisions as a single change. Schema changes
  require an inspected revision in `alembic/versions/`.
- Generate with `uv run alembic revision --autogenerate -m "..."`, then review
  both `upgrade()` and `downgrade()` manually. Pay special attention to SQLite
  batch operations, defaults, nullability, indexes, and existing rows.
- Validate with `uv run alembic upgrade head` and relevant ORM tests. Exercise a
  downgrade when it is safe and meaningful.
- Do not edit an already deployed migration to change history. Add a corrective
  migration instead.
- Never commit local SQLite, PostgreSQL, Chroma, or Milvus data files.

### Authentication, authorization, and audit

- Okta identity and group claims are untrusted input until validated. Keep group
  mapping centralized and default to no additional privilege.
- Server/data-layer authorization must back every UI restriction. Hiding a page,
  button, or column is not sufficient authorization.
- Preserve JIT user synchronization, logout/session cleanup, stale-cookie
  handling, and audit events when changing authentication.
- Deployments sharing a hostname but using different paths/databases must have
  distinct `cookie.prefix` values; otherwise one deployment can read another's
  user cookie and resolve the wrong local user ID.
- Audit records are append-oriented evidence. Avoid destructive rewrites and do
  not record secrets, tokens, full prompts with PHI, or raw result sets unless an
  approved logging policy explicitly requires them.

## Test design

- Start with the smallest test that proves the behavior, then run the nearest
  module/package suite and a broader suite proportional to risk.
- Make tests hermetic: use `tmp_path`, in-memory/temporary databases, monkeypatch,
  and deterministic fixtures. Do not depend on repository-local databases,
  developer state, test ordering, or live services.
- Time-window tests must derive ordinary records from the current/test clock.
  Tests for expiry boundaries should create explicit recent and expired values.
- Assert meaningful outcomes and failure modes, not incidental implementation
  details. Avoid snapshots that conceal sensitive values or accept broad churn.
- A failing full suite is not automatically caused by the current change.
  Reproduce failures in isolation, document the evidence, and create a separate
  issue for unrelated defects rather than expanding scope silently.
- Do not remove, skip, or relax a test merely to obtain a green run. Explain
  environment-dependent skips in the PR.

## Security, privacy, and repository hygiene

- Never commit credentials, cookies, tokens, `.streamlit/secrets.toml`, `.env`
  files, connection strings, or production configuration.
- Never commit PHI, patient identifiers, rosters, prompts/results containing
  clinical details, screenshots, evaluation outputs, local databases, or logs
  derived from real users. Treat filenames and metadata as potentially sensitive.
- Use synthetic or irreversibly de-identified fixtures. Do not copy production
  rows into a test and "scrub them later."
- Inspect staged files and the staged diff before every commit. In a mixed
  worktree, preserve all unrelated tracked and untracked user files.
- Do not run destructive database, deployment, git, or cleanup commands without
  explicit authorization and a clear recovery path.
- Production and dev deployment credentials live outside source control. The
  `dev` branch deployment is automated; merging is a deployment-affecting action.

## Documentation and comments

- Update nearby documentation when behavior, configuration, commands, or
  ownership changes. Prefer links to the source of truth over duplicating it.
- Comments should explain non-obvious constraints and decisions, not restate the
  code. Preserve references to issues or partner decisions when they explain why
  a guardrail exists.
- Architecture documentation must distinguish current behavior, proposed
  behavior, and unresolved partner decisions.

## Definition of done

A change is done only when all applicable items are true:

- the issue scope and ownership are correct;
- implementation and migrations are complete and narrowly scoped;
- focused tests pass and broader verification is recorded;
- Ruff/pre-commit checks pass for changed Python files;
- no credentials, PHI, local artifacts, or unrelated changes are staged;
- security, authorization, audit, and rollback impact were considered;
- documentation/configuration examples match the implementation;
- the PR is reviewed and merged into the intended base branch;
- completed issues are closed with evidence and Project cards are current;
- remaining HeL or joint work is explicit, assigned/labeled accurately, and left
  open.

In the final handoff, report the branch/PR, files changed, exact validation
results, unresolved risks or external dependencies, and any local user changes
that were deliberately left untouched.
