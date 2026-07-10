# Authenticated Evaluation Workspace Design

**Date:** 2026-07-10
**Status:** Revised design pending final spec approval
**Scope:** Agentic answer feedback plus evaluation launched and reviewed through Thrive's authenticated UI

## Context

The existing agentic evaluation harness runs a curated question-by-patient matrix, records tool and latency evidence, applies an LLM judge for triage, and generates a self-contained HTML report for human review. It is currently a developer-operated CLI workflow: an operator prepares YAML, starts the run, generates an HTML file, and opens it locally.

Thrive also records thumbs-up and thumbs-down feedback on production answers. A thumbs-down interaction can include a category and free-text explanation, but that evidence is not integrated with the evaluation harness.

That feedback UI currently belongs to the legacy Vanna summary path. Completed agentic answers do not expose equivalent thumbs-up or thumbs-down controls, so the application cannot yet produce the agentic feedback cases this workspace needs. Agentic feedback capture is therefore part of this feature, not an external prerequisite.

The July 7, 2026 Thrive discussion connected these needs. The team wanted to test natural questions asked by specific users, reduce manual transcription of failures, preserve human oversight of scoring, and make long-running reports easier to launch and inspect. A completed browser report cannot reliably "pop open" from a Streamlit session that may no longer exist. Long evaluations also should not block or depend on a browser request.

## Goals

1. Add thumbs-up and thumbs-down feedback to completed agentic answers.
2. Give admins one authenticated workspace for feedback-derived and curated evaluations.
3. Replay one thumbs-down interaction exactly, with the same patient and relevant conversation context.
4. Support promotion of a useful feedback case into a reusable curated case.
5. Run one feedback replay synchronously and larger selections asynchronously.
6. Keep progress, results, and review behind Thrive authentication.
7. Preserve the current model in which the LLM judge provides triage and a human supplies the final verdict.
8. Reuse the existing harness's runner, collection, judge, latency, and developer-report capabilities.
9. Apply the existing agent-log fidelity and retention policy to evaluation payloads.

## Non-goals

- Automatically opening a browser tab after an unattended run.
- Publishing unauthenticated or externally hosted PHI-bearing reports.
- Treating the LLM judge as the authoritative pass/fail decision.
- Running competing answers side by side in the chat experience.
- Deploying Redis, Celery, or another external queue in the first version.
- Allowing production web requests to edit repository YAML files.
- Generalizing the evaluation workspace to non-agentic legacy chat in the first version.
- Sending agentic thumbs-up feedback into Vanna's single-question/single-SQL training workflow.

## Chosen approach

Build a unified authenticated evaluation workspace with a durable application model and a lightweight worker. Streamlit creates and reviews evaluation runs; an evaluation service executes normalized cases; authenticated Thrive pages render status and results.

The worker boundary must be replaceable by an external queue later without changing the case, run, result, or Admin UI contracts.

The existing CLI remains available for developers. It uses the same normalized case execution contracts and may continue producing a local self-contained HTML report when explicitly invoked, but JSON and HTML files are not the application's storage or security boundary.

## Architecture

### Agentic response feedback capture

Every completed agentic user turn presents one feedback control set for the final visible answer:

- thumbs up;
- thumbs down;
- a required category and optional free-text explanation for thumbs down.

The initial agentic thumbs-down categories are: incorrect answer, incomplete or missing information, wrong patient or conversation context, wrong data or tool use, response too slow, did not understand the question, and other. These are agent-oriented product signals rather than the Vanna flow's SQL-training taxonomy.

The controls render once for the completed answer group, not on intermediate narration, thinking, tool-call cards, SQL artifacts, charts, or dataframes. A failed or interrupted run may expose thumbs down only when a final user-visible answer exists.

Feedback targets the `AgentRun`, which already represents one agentic question/turn and links the originating user message, final answer, selected patient, model, tool evidence, and final persisted message. It does not target an arbitrary `MessageType.TEXT` row because one agentic answer may span several persisted messages.

The application stores agentic feedback independently from the legacy `Message.feedback` training behavior. A thumbs up records positive product feedback only; it does not create a Vanna question/SQL training pair. A user may change or clear their feedback, and each change is auditable.

If an agentic run invokes the Vanna fallback, the answer group still exposes only one feedback control set. The feedback remains linked to the initiating `AgentRun`, whose fallback metadata identifies how the visible answer was produced.

### Evaluation catalog

The Admin **Evaluations** section presents two sources in one launch screen:

- **Thumbs-down feedback:** filterable by user, organization, patient, date, feedback category, and question.
- **Curated suite:** the full suite or selected curated cases and patients.

The page shows selected case and patient counts, the resulting conversation and turn counts, expected execution mode, and a warning about model and warehouse usage.

Authorization is enforced in both the UI and service layers. Only admins may launch runs, inspect results, promote cases, or record verdicts.

### Case snapshot service

The snapshot service converts a selected production interaction into an immutable feedback evaluation case. It captures:

- originating message and agent-run identifiers;
- user and organization identity;
- selected patient/source identity;
- exact question and relevant preceding conversation turns;
- original answer and available tool-call evidence;
- thumbs-down category and free-text explanation;
- snapshot creator and time.

A snapshot prevents later message edits, feedback changes, or source-log cleanup from silently changing the test represented by an existing run.

Curated YAML cases are normalized into the same internal case interface. Their parameterized questions, patient inputs, follow-ups, and reviewer guidance remain supported.

### Evaluation runner

The evaluation runner reuses the agentic runner, event collector, latency attribution, and LLM judge. It accepts normalized cases and emits normalized case results independent of whether the caller is the Admin UI or CLI.

Exactly one feedback interaction uses the synchronous path. Any selection larger than one case—including feedback batches, curated subsets, and the full suite—uses the asynchronous path.

### Authenticated results workspace

Results are rendered from authenticated application records rather than public or directly linked report files.

For feedback cases, the workspace displays:

- the original thumbs-down explanation;
- original and rerun answers side by side;
- each answer's tool, SQL, reliability, and timing evidence, subject to logging mode;
- LLM triage, rationale, and confidence;
- the admin's separate final verdict and note;
- a **Promote to suite** action.

For curated cases, the workspace preserves the existing suite-oriented scorecard, conversation, tool, latency, and review concepts.

Asynchronous completion creates a persistent in-app Admin notification containing no PHI and a **View report** action.

## Data model

### Evaluation case

An evaluation case contains:

- source type: `feedback` or `curated`;
- immutable case version;
- source message/run identifiers when applicable;
- originating user and organization;
- patient/source identity snapshot;
- question and relevant conversation context;
- original response and evidence for feedback cases;
- feedback category and comment;
- reviewer guidance;
- creator and timestamps;
- optional relationship to a promoted curated case.

### Agentic run feedback

An agentic feedback record is keyed to the `AgentRun` and contains:

- run ID and originating user ID;
- rating: up or down;
- required thumbs-down category when the rating is down;
- optional free-text explanation;
- created and updated timestamps.

The current record supports efficient chat rendering and Admin filtering. An append-only audit event records prior and new values whenever feedback changes or is cleared. The evaluation catalog reads agentic feedback through this run-level relationship; it does not infer feedback by scanning the run's individual assistant messages.

### Evaluation run

A run contains:

- run type: single feedback, feedback batch, curated subset, or full suite;
- immutable references to selected case versions;
- requesting admin;
- synchronous or asynchronous execution mode;
- lifecycle state: queued, running, completed, completed-with-errors, failed, cancelled, or interrupted;
- total, completed, failed, and pending case counts;
- created, started, heartbeat, and completed timestamps;
- model/provider and relevant agent configuration snapshot;
- aggregate latency and verdict totals.

### Case result

A case result contains:

- run and case-version references;
- attempt number and execution state;
- rerun answer and streamed events;
- tool calls, SQL evidence, reliability notes, and timing;
- execution error, when present;
- LLM triage verdict, rationale, and confidence;
- final admin verdict: correct, incorrect, or can't tell;
- admin review note, reviewer identity, and review time;
- comparison status against the original feedback concern.

### Notification and audit history

Persistent Admin notifications reference completed runs without including patient names, questions, answers, or other PHI.

Audit history records who launched, cancelled, resumed, promoted, and reviewed each run or case. Verdict changes append history rather than silently replacing the prior decision.

## Promotion to the curated suite

**Promote to suite** creates a draft curated case from a feedback snapshot. The exact feedback replay remains unchanged for historical comparison.

The draft may generalize patient-specific wording into parameters. Before activation, an admin must supply or confirm:

- reusable prompt and follow-up structure;
- patient-selection requirements;
- expected behavior;
- reviewer guidance.

The production application stores the draft and its approval state. It does not edit deployed source files during a web request. Exporting or synchronizing approved definitions into repository YAML occurs through a controlled development workflow.

## Execution flow

### Single synchronous feedback replay

1. The admin selects exactly one thumbs-down interaction.
2. Thrive validates access and snapshots the interaction, patient, and relevant conversation context.
3. Thrive creates the run and case-result records.
4. The evaluation runner executes the case through the agentic pipeline.
5. The judge receives the rerun output, tool evidence, and original thumbs-down explanation as reviewer guidance.
6. Thrive commits the result and navigates to the authenticated comparison view.

If the request disconnects, the durable run remains discoverable. Thrive preserves any committed result and does not silently retry or spawn an unmanaged browser process.

### Asynchronous batch or suite run

1. Thrive validates and snapshots every selected case before queueing.
2. It creates the run and result placeholders atomically.
3. The lightweight worker claims the oldest eligible run.
4. It executes cases sequentially and updates heartbeat and progress fields.
5. Each case result is committed independently.
6. Case-level model or warehouse failures are recorded, and safe sibling cases continue.
7. Run-level initialization or authorization failures fail the run before case execution.
8. Completion records aggregate status and creates a persistent Admin notification.

The first version permits one active asynchronous evaluation run per deployment. Additional runs remain queued. This protects the shared model and warehouse from the contention observed during the meeting.

### Recovery and retries

A run whose heartbeat expires becomes interrupted. An admin may resume only unstarted or failed cases. Completed cases are not rerun during resume. Explicitly starting a new run is required to generate a new result for a completed case.

Attempts are numbered and retained for auditability. A queued run may be cancelled before execution. A running run stops after its current case when cancellation is requested.

## Security and privacy

- Admin authorization is mandatory at every route and service boundary.
- Only the user who owns an agentic run may create, change, or clear its chat feedback; admins inspect that feedback through the authenticated evaluation and audit surfaces rather than submitting feedback on the user's behalf.
- Selection is limited to interactions accessible to the current deployment and database.
- Saved identifiers never authorize cross-deployment lookup.
- Reports are rendered through authenticated Thrive pages; no public report URLs are produced.
- Notifications, worker logs, process titles, and command-line arguments contain run identifiers rather than serialized PHI.
- Patient display follows the same rules as existing Admin pages.
- Evaluation snapshots, results, judge rationale, and tool evidence follow the configured agent-log fidelity and retention policy.

Logging-mode behavior is explicit:

- `full`: complete permitted evidence is retained.
- `scrubbed`: existing SQL literal and result-row protections apply.
- `disabled`: Thrive rejects launches requiring unavailable evidence and explains why; it does not pretend an exact replay can be reconstructed.

When retention removes sensitive payloads, minimal non-PHI run metadata may remain to explain that details expired.

## Failure handling

- **Snapshot failure:** reject the affected case before launch and identify the missing required context.
- **Patient resolution failure:** fail closed rather than substituting another patient.
- **Model or warehouse failure:** record the case error, preserve completed cases, and continue a batch when safe.
- **Judge failure:** retain the agent result and mark LLM triage unavailable; human review remains possible.
- **Worker crash:** heartbeat expiry marks the run interrupted and unfinished cases resumable.
- **Expired unsnapshotted evidence:** explain that exact replay is unavailable.
- **Synchronous timeout or disconnect:** preserve the durable run and direct the admin to its run page when possible.
- **Concurrent demand:** queue runs behind the single active asynchronous run and show queue status.

## User interface

The chat experience adds:

1. One compact thumbs-up/thumbs-down control set after each completed agentic answer group.
2. A thumbs-down popover with the agentic category list and an optional 500-character explanation.
3. Submitted-state feedback and the ability for the originating user to change or clear it.
4. No feedback controls on intermediate agent artifacts.

The approved Admin workspace uses:

1. An **Evaluations** navigation destination.
2. A source switcher for **Thumbs-down feedback** and **Curated suite**.
3. Source-appropriate filters and selectable case rows.
4. A launch summary showing scope and execution mode.
5. A comparison result with the feedback concern above original and rerun answers.
6. Separate LLM triage and admin-final-verdict controls.
7. A **Promote to suite** action on feedback results.
8. A run list with queued, active, interrupted, and completed states.
9. Persistent PHI-free completion notifications linking to authenticated results.

## Testing strategy

### Unit tests

Cover:

- one feedback control set on the completed agentic answer group;
- no controls on intermediate narration, thinking, tools, or artifacts;
- agentic feedback persistence, update, and clear behavior;
- rejection of feedback writes from a user who does not own the run;
- no Vanna training write on agentic thumbs up;
- one run-level feedback target when the agentic flow invokes Vanna fallback;
- feedback interaction to immutable snapshot;
- curated YAML to normalized case;
- patient and conversation-context reconstruction;
- execution-mode selection: one feedback case synchronous, all larger selections asynchronous;
- run and result lifecycle transitions;
- heartbeat expiry and resume selection;
- separation of judge triage and final human verdict;
- fidelity and retention policy application;
- service-layer admin authorization;
- draft promotion without source-file mutation.

### Integration tests

Use fake agent, judge, and analytics adapters to verify:

- single replay from launch through authenticated comparison;
- completed agentic answer to thumbs-down Admin catalog entry;
- independent commits for batch case results;
- case failure without loss of successful siblings;
- resume of unfinished cases only;
- PHI-free persistent completion notification;
- rejection of non-admin launch and report access;
- scrubbed logging protections;
- clean rejection when disabled logging prevents exact replay.

### CLI compatibility tests

Verify that the developer harness continues to:

- load the current YAML suite and roster;
- run a selected subset or full matrix;
- generate the explicit local self-contained HTML report;
- use the same normalized execution and result contracts as the Admin workflow.

## Acceptance criteria

The feature is complete when:

1. A user receives one thumbs-up/thumbs-down control set on a completed agentic answer, with category and explanation capture for thumbs down.
2. Agentic thumbs up records product feedback without writing to Vanna training data.
3. An admin can filter agentic thumbs-down interactions by user and select one.
4. An admin can rerun it synchronously with the same patient and relevant conversation context.
5. An admin can compare original and rerun answers inside authenticated Thrive.
6. An admin can see LLM triage and submit a separate final human verdict.
7. An admin can promote the interaction into a draft reusable case without editing deployed source files.
8. An admin can select multiple feedback cases or the full curated suite.
9. An admin can leave the page while an asynchronous run continues.
10. An admin can return to a persistent completion notification and open the authenticated report.
11. An admin can inspect partial results and safely resume unfinished cases after interruption.
12. A non-admin cannot launch, inspect, or retrieve evaluation data.

## Future extension points

- Replace the lightweight worker with an external queue implementation.
- Add scheduled or release-triggered suite runs.
- Add configurable concurrency after production capacity is measured.
- Add controlled repository synchronization for approved curated cases.
- Add organization-scoped evaluator roles if access expands beyond admins.
