# Authenticated Evaluation Workspace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add thumbs feedback to completed agentic answers and give admins a durable, authenticated workspace for launching, monitoring, and reviewing feedback-derived and curated agent evaluations.

**Architecture:** Persist agentic feedback and evaluation state in the app SQLite database, normalize feedback snapshots and YAML cases behind one case interface, and extract the existing CLI harness loop into a reusable executor. A process-local worker claims durable queued runs through a replaceable queue interface; Streamlit only creates runs and renders authenticated state.

**Tech Stack:** Python 3.13, Streamlit, SQLAlchemy 2, Alembic, Pydantic, pydantic-ai, pytest, SQLite application state, existing `agent/` runner and `evals/` harness modules.

## Global Constraints

- Admin authorization is enforced in the UI and every evaluation service mutation/read.
- Only the originating user may create, change, or clear feedback on an agentic run.
- Render one feedback control set per completed agentic answer group; never render controls on thinking, narration, tool calls, SQL, charts, dataframes, or chooser cards.
- Agentic thumbs up records product feedback only and never invokes Vanna training.
- One feedback case runs synchronously; every curated run and every selection of two or more cases runs asynchronously.
- Permit one active asynchronous evaluation run per deployment.
- Persist PHI-bearing snapshots and results only in authenticated application records and apply `[agent_logging]` fidelity and retention.
- Notifications contain no patient name, question, answer, or tool evidence.
- Production requests never edit `evals/questions.yaml`.
- LLM judgment is triage; `correct`, `incorrect`, or `cant_tell` from an admin is authoritative.
- Standalone JSON/HTML output remains an explicit developer CLI capability, not the production report boundary.
- No new queue dependency is introduced; the queue/executor interfaces must permit a later external worker.

---

## File Structure

- `orm/evaluation_models.py`: feedback, immutable cases, runs, results, review events, and Admin notifications.
- `orm/evaluation_functions.py`: ownership-checked feedback operations and admin-checked evaluation CRUD/query operations.
- `evals/cases.py`: source-neutral case dataclasses, YAML normalization, and feedback snapshot construction.
- `evals/executor.py`: reusable agent/judge execution of one normalized conversation and one durable run.
- `evals/worker.py`: queue protocol, SQLite claim implementation, heartbeat/recovery, and process-local worker lifecycle.
- `views/agent_feedback.py`: Streamlit feedback controls for a completed agentic answer group.
- `views/admin_evaluations.py`: Admin catalog, launch, progress, comparison, verdict, promotion, and notification rendering.
- `scripts/agent_eval_harness.py`: CLI adapter over `evals.executor`, retaining JSON output.
- `app.py`: starts the singleton process-local worker after migrations.
- `views/admin.py`: adds the Evaluations Admin tab.

---

### Task 1: Evaluation and Agentic Feedback Schema

**Files:**
- Create: `orm/evaluation_models.py`
- Modify: `orm/models.py:570-642,835-858`
- Create: `alembic/versions/d91c7a4ef201_add_authenticated_evaluations.py`
- Create: `tests/orm/test_evaluation_models.py`
- Create: `tests/orm/test_authenticated_evaluation_migration.py`

**Interfaces:**
- Consumes: existing `orm.models.Base`, `AgentRun`, `Message`, and `User`.
- Produces: `AgentRunFeedback`, `AgentRunFeedbackEvent`, `EvaluationCase`, `EvaluationRun`, `EvaluationCaseResult`, `EvaluationReviewEvent`, and `AdminNotification` ORM classes.

- [ ] **Step 1: Write model round-trip tests**

Create `tests/orm/test_evaluation_models.py` with an in-memory SQLite fixture and these assertions:

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from orm.models import AgentRun, Base, RoleTypeEnum, User, UserRole
from orm.evaluation_models import (
    AdminNotification,
    AgentRunFeedback,
    AgentRunFeedbackEvent,
    EvaluationCase,
    EvaluationCaseResult,
    EvaluationReviewEvent,
    EvaluationRun,
)


def _session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _seed_user_and_run(session):
    role = UserRole(role_name="Doctor", description="Doctor", role=RoleTypeEnum.DOCTOR)
    session.add(role)
    session.flush()
    user = User(
        username="doctor",
        first_name="Test",
        last_name="Doctor",
        password="x",
        email="doctor@example.test",
        organization="Thrive",
        user_role_id=role.id,
    )
    session.add(user)
    session.flush()
    run = AgentRun(
        run_id="agent-run-1",
        session_id="session-1",
        group_id="group-1",
        user_id=user.id,
        user_role=1,
        status="success",
        success=True,
        logging_mode="full",
    )
    session.add(run)
    session.commit()
    return user, run


def test_feedback_and_audit_event_round_trip():
    session = _session()
    user, run = _seed_user_and_run(session)
    feedback = AgentRunFeedback(
        agent_run_id=run.id,
        user_id=user.id,
        rating="down",
        category="incorrect_answer",
        comment="The medication status is wrong.",
    )
    session.add(feedback)
    session.flush()
    session.add(
        AgentRunFeedbackEvent(
            feedback_id=feedback.id,
            actor_user_id=user.id,
            old_rating=None,
            new_rating="down",
            old_category=None,
            new_category="incorrect_answer",
            old_comment=None,
            new_comment="The medication status is wrong.",
        )
    )
    session.commit()
    assert session.query(AgentRunFeedback).one().agent_run_id == run.id
    assert session.query(AgentRunFeedbackEvent).one().new_rating == "down"


def test_case_run_result_review_and_notification_round_trip():
    session = _session()
    user, _ = _seed_user_and_run(session)
    case = EvaluationCase(
        case_id="case-1",
        version=1,
        source_type="feedback",
        status="active",
        payload_json='{"question":"Which medications are active?"}',
        created_by=user.id,
    )
    run = EvaluationRun(
        run_id="eval-run-1",
        run_type="single_feedback",
        execution_mode="synchronous",
        status="queued",
        requested_by=user.id,
        total_cases=1,
    )
    session.add_all([case, run])
    session.flush()
    result = EvaluationCaseResult(
        evaluation_run_id=run.id,
        evaluation_case_id=case.id,
        ordinal=0,
        attempt=1,
        status="pending",
    )
    session.add(result)
    session.flush()
    session.add(
        EvaluationReviewEvent(
            result_id=result.id,
            reviewer_id=user.id,
            old_verdict=None,
            new_verdict="cant_tell",
            note="Needs clinical review.",
        )
    )
    session.add(
        AdminNotification(
            user_id=user.id,
            evaluation_run_id=run.id,
            kind="evaluation_completed",
        )
    )
    session.commit()
    assert session.query(EvaluationCaseResult).one().attempt == 1
    assert session.query(EvaluationReviewEvent).one().new_verdict == "cant_tell"
    assert session.query(AdminNotification).one().read_at is None
```

- [ ] **Step 2: Run the model tests and verify they fail**

Run: `uv run pytest tests/orm/test_evaluation_models.py -q`

Expected: collection fails because `orm.evaluation_models` does not exist.

- [ ] **Step 3: Define the focused ORM models**

Create `orm/evaluation_models.py`. Use `Base` from `orm.models`; use `String(36)` public IDs, integer internal foreign keys, `Text` JSON payloads, database timestamps, and these constraints:

```python
class AgentRunFeedback(Base):
    __tablename__ = "thrive_agent_run_feedback"
    __table_args__ = (
        UniqueConstraint("agent_run_id", "user_id", name="uq_agent_run_feedback_owner"),
        CheckConstraint("rating IN ('up', 'down')", name="ck_agent_run_feedback_rating"),
    )
    id = Column(Integer, primary_key=True)
    agent_run_id = Column(Integer, ForeignKey("thrive_agent_run.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(Integer, ForeignKey("thrive_user.id", ondelete="CASCADE"), nullable=False)
    rating = Column(String(8), nullable=False)
    category = Column(String(64), nullable=True)
    comment = Column(String(500), nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now(), nullable=False)
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now(), nullable=False)


class AgentRunFeedbackEvent(Base):
    __tablename__ = "thrive_agent_run_feedback_event"
    id = Column(Integer, primary_key=True)
    feedback_id = Column(Integer, nullable=False)
    actor_user_id = Column(Integer, ForeignKey("thrive_user.id"), nullable=False)
    old_rating = Column(String(8), nullable=True)
    new_rating = Column(String(8), nullable=True)
    old_category = Column(String(64), nullable=True)
    new_category = Column(String(64), nullable=True)
    old_comment = Column(String(500), nullable=True)
    new_comment = Column(String(500), nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now(), nullable=False)
```

Define the remaining classes with these exact SQLAlchemy declarations:

```python
class EvaluationCase(Base):
    __tablename__ = "thrive_evaluation_case"
    __table_args__ = (UniqueConstraint("case_id", "version", name="uq_evaluation_case_version"),)
    id = Column(Integer, primary_key=True)
    case_id = Column(String(36), nullable=False)
    version = Column(Integer, nullable=False)
    source_type = Column(String(16), nullable=False)
    status = Column(String(16), nullable=False)
    source_feedback_id = Column(Integer, ForeignKey("thrive_agent_run_feedback.id"), nullable=True)
    source_message_id = Column(Integer, ForeignKey("thrive_message.id"), nullable=True)
    source_agent_run_id = Column(Integer, ForeignKey("thrive_agent_run.id"), nullable=True)
    promoted_from_case_id = Column(Integer, ForeignKey("thrive_evaluation_case.id"), nullable=True)
    payload_json = Column(Text, nullable=False)
    created_by = Column(Integer, ForeignKey("thrive_user.id"), nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now(), nullable=False)
    expires_at = Column(TIMESTAMP, nullable=True)

class EvaluationRun(Base):
    __tablename__ = "thrive_evaluation_run"
    id = Column(Integer, primary_key=True)
    run_id = Column(String(36), nullable=False, unique=True)
    run_type = Column(String(24), nullable=False)
    execution_mode = Column(String(16), nullable=False)
    status = Column(String(32), nullable=False)
    requested_by = Column(Integer, ForeignKey("thrive_user.id"), nullable=False)
    total_cases = Column(Integer, nullable=False)
    completed_cases = Column(Integer, nullable=False, default=0)
    failed_cases = Column(Integer, nullable=False, default=0)
    cancel_requested = Column(Boolean, nullable=False, default=False)
    model_json = Column(Text, nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now(), nullable=False)
    started_at = Column(TIMESTAMP, nullable=True)
    heartbeat_at = Column(TIMESTAMP, nullable=True)
    completed_at = Column(TIMESTAMP, nullable=True)

class EvaluationCaseResult(Base):
    __tablename__ = "thrive_evaluation_case_result"
    __table_args__ = (UniqueConstraint("evaluation_run_id", "evaluation_case_id", "attempt", name="uq_eval_result_attempt"),)
    id = Column(Integer, primary_key=True)
    evaluation_run_id = Column(Integer, ForeignKey("thrive_evaluation_run.id", ondelete="CASCADE"), nullable=False)
    evaluation_case_id = Column(Integer, ForeignKey("thrive_evaluation_case.id"), nullable=False)
    ordinal = Column(Integer, nullable=False)
    attempt = Column(Integer, nullable=False)
    status = Column(String(24), nullable=False)
    result_json = Column(Text, nullable=True)
    error_type = Column(String(100), nullable=True)
    error_message = Column(String(500), nullable=True)
    judge_verdict = Column(String(24), nullable=True)
    judge_reason = Column(String(500), nullable=True)
    judge_confidence = Column(Float, nullable=True)
    final_verdict = Column(String(24), nullable=True)
    review_note = Column(String(1000), nullable=True)
    reviewed_by = Column(Integer, ForeignKey("thrive_user.id"), nullable=True)
    reviewed_at = Column(TIMESTAMP, nullable=True)
    started_at = Column(TIMESTAMP, nullable=True)
    completed_at = Column(TIMESTAMP, nullable=True)

class EvaluationReviewEvent(Base):
    __tablename__ = "thrive_evaluation_review_event"
    id = Column(Integer, primary_key=True)
    result_id = Column(Integer, ForeignKey("thrive_evaluation_case_result.id", ondelete="CASCADE"), nullable=False)
    reviewer_id = Column(Integer, ForeignKey("thrive_user.id"), nullable=False)
    old_verdict = Column(String(24), nullable=True)
    new_verdict = Column(String(24), nullable=False)
    note = Column(String(1000), nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now(), nullable=False)

class AdminNotification(Base):
    __tablename__ = "thrive_admin_notification"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("thrive_user.id", ondelete="CASCADE"), nullable=False)
    evaluation_run_id = Column(Integer, ForeignKey("thrive_evaluation_run.id", ondelete="CASCADE"), nullable=False)
    kind = Column(String(40), nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now(), nullable=False)
    read_at = Column(TIMESTAMP, nullable=True)
```

Add indexes on every foreign key used by catalog, run, and notification queries. Import the classes at the end of `orm/models.py` so `Base.metadata` and Alembic discover them.

- [ ] **Step 4: Add the Alembic migration and migration tests**

Create revision `d91c7a4ef201`, with `down_revision = "c233fb500001"`. Create the seven tables and indexes above in dependency order; downgrade them in reverse order. In `tests/orm/test_authenticated_evaluation_migration.py`, upgrade a temporary SQLite DB to `head`, assert all seven tables exist, downgrade to `c233fb500001`, and assert all seven are absent.

- [ ] **Step 5: Run schema tests**

Run: `uv run pytest tests/orm/test_evaluation_models.py tests/orm/test_authenticated_evaluation_migration.py -q`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add orm/models.py orm/evaluation_models.py alembic/versions/d91c7a4ef201_add_authenticated_evaluations.py tests/orm/test_evaluation_models.py tests/orm/test_authenticated_evaluation_migration.py
git commit -m "feat(evals): add durable evaluation schema"
```

---

### Task 2: Ownership-Checked Agentic Feedback Service

**Files:**
- Create: `orm/evaluation_functions.py`
- Create: `tests/orm/test_agent_run_feedback_functions.py`
- Modify: `agent/deps_builder.py:127-166`
- Modify: `agent/run_logger.py:1-16,130-190,323-370`
- Modify: `tests/agent/test_deps_builder.py`
- Modify: `tests/agent/test_run_logger.py`

**Interfaces:**
- Produces: `get_agent_feedback(group_id: str, user_id: int) -> AgentFeedbackView | None`, `set_agent_feedback(run_id: str, actor_user_id: int, rating: Literal['up','down'], category: str | None, comment: str | None) -> AgentFeedbackView`, `clear_agent_feedback(run_id: str, actor_user_id: int) -> None`, and `link_agent_final_message(run_id: str, message_id: int) -> None`.
- Produces: a minimal `AgentRun` envelope even when `[agent_logging].mode = "disabled"`; detailed events and answer payloads remain disabled.

- [ ] **Step 1: Write failing ownership and validation tests**

Create tests that seed two users and one `AgentRun`; verify:

```python
saved = set_agent_feedback("run-1", owner.id, "down", "wrong_patient_context", "Wrong chart")
assert saved.rating == "down"
assert saved.category == "wrong_patient_context"
assert count_feedback_events() == 1

with pytest.raises(PermissionError):
    set_agent_feedback("run-1", other.id, "up", None, None)

with pytest.raises(ValueError, match="category"):
    set_agent_feedback("run-1", owner.id, "down", None, None)

set_agent_feedback("run-1", owner.id, "up", None, None)
assert current_feedback().rating == "up"
assert count_feedback_events() == 2

clear_agent_feedback("run-1", owner.id)
assert current_feedback() is None
assert latest_feedback_event().new_rating is None
```

Also patch `write_to_file_and_training` and assert it is never imported or called by `set_agent_feedback`.

- [ ] **Step 2: Run the service tests and verify they fail**

Run: `uv run pytest tests/orm/test_agent_run_feedback_functions.py -q`

Expected: import failure for `orm.evaluation_functions`.

- [ ] **Step 3: Implement the feedback service**

In `orm/evaluation_functions.py`, define:

```python
AGENT_FEEDBACK_CATEGORIES = frozenset({
    "incorrect_answer",
    "incomplete_information",
    "wrong_patient_context",
    "wrong_data_or_tool",
    "response_too_slow",
    "did_not_understand",
    "other",
})

@dataclass(frozen=True)
class AgentFeedbackView:
    run_id: str
    rating: Literal["up", "down"]
    category: str | None
    comment: str | None


def set_agent_feedback(run_id, actor_user_id, rating, category=None, comment=None):
    if rating not in {"up", "down"}:
        raise ValueError("rating must be 'up' or 'down'")
    if rating == "down" and category not in AGENT_FEEDBACK_CATEGORIES:
        raise ValueError("thumbs-down feedback requires a valid category")
    if rating == "up":
        category = comment = None
    if comment is not None and len(comment) > 500:
        raise ValueError("comment must be 500 characters or fewer")
    # Load AgentRun by run_id, require run.user_id == actor_user_id,
    # upsert AgentRunFeedback, append AgentRunFeedbackEvent, commit, return view.
```

Use a single transaction for current state plus audit event. `clear_agent_feedback` appends the clear event before deleting the current row. Keep this module Streamlit-free.

- [ ] **Step 4: Preserve a minimal run envelope in disabled logging mode**

Change `build_agent_deps` to construct `AgentRunLogger` for all modes. In `AgentRunLogger.start_run`, always insert the run identity, ownership, group, and message linkage; when `config.mode == "disabled"`, set `question`, patient fields, history, provider/model, and hashes to `None`, then return without an event. In every detailed event/tool method, return immediately for disabled mode. In `finalize_run`, update status and `final_message_id` for disabled mode but leave `final_answer_text`, usage, error, and stack trace unset.

This gives feedback a stable run subject without turning disabled logging back into PHI-bearing agent logging.

- [ ] **Step 5: Run feedback and logger tests**

Run: `uv run pytest tests/orm/test_agent_run_feedback_functions.py tests/agent/test_deps_builder.py tests/agent/test_run_logger.py -q`

Expected: all tests pass, including an assertion that disabled mode creates one minimal `AgentRun` and zero `AgentRunEvent` rows.

- [ ] **Step 6: Commit**

```bash
git add orm/evaluation_functions.py agent/deps_builder.py agent/run_logger.py tests/orm/test_agent_run_feedback_functions.py tests/agent/test_deps_builder.py tests/agent/test_run_logger.py
git commit -m "feat(agent): persist owned run feedback"
```

---

### Task 3: Agentic Answer Feedback Controls

**Files:**
- Create: `views/agent_feedback.py`
- Modify: `agent/runtime.py:403-587`
- Modify: `utils/chat_bot_helper.py:771-803,1248-1272`
- Create: `tests/views/test_agent_feedback.py`
- Modify: `tests/utils/test_message_grouping.py`
- Create: `tests/agent/test_runtime_agent_feedback.py`

**Interfaces:**
- Consumes: Task 2 feedback functions.
- Produces: `render_agent_feedback(group_id: str, user_id: int) -> None` and reliable `AgentRun.final_message_id` linkage.

- [ ] **Step 1: Write failing rendering tests**

Test that `render_message_group` calls `render_agent_feedback` exactly once after an agentic group and never for:

- a Vanna group without an `AgentRun`;
- a group whose run is still open;
- a group with only an error and no final answer;
- individual thinking, tool, chart, dataframe, and chooser messages.

Mock the service view and assert the selected thumbs button uses `type="primary"`. Submit down feedback with `wrong_data_or_tool` and a 501-character comment; assert the callback rejects the comment without writing. Submit a valid 500-character comment; assert the service receives it.

- [ ] **Step 2: Run UI tests and verify they fail**

Run: `uv run pytest tests/views/test_agent_feedback.py tests/utils/test_message_grouping.py tests/agent/test_runtime_agent_feedback.py -q`

Expected: failure because `views.agent_feedback` and final-message linking are absent.

- [ ] **Step 3: Make final message persistence return and link the message**

Return the saved `Message` from `add_message`. In `agent/runtime.py`, record `state["last_persisted_text_message_id"]` for `AssistantTextCompletedEvent`. For `FinalResponseEvent`, use the newly saved text message ID or the deduplicated streamed message ID, then call:

```python
run_id = st.session_state.get("agent_current_run_id")
if run_id and final_message_id:
    link_agent_final_message(run_id, final_message_id)
```

The linking function updates only `final_message_id`; it does not copy answer text into disabled logs.

- [ ] **Step 4: Implement the Streamlit control**

In `views/agent_feedback.py`, map display labels to Task 2 category values and render:

```python
cols = st.columns([0.08, 0.08, 0.84])
with cols[0]:
    st.button("👍", key=f"agent_up_{run_id}", type="primary" if rating == "up" else "secondary", on_click=_save_up)
with cols[1]:
    with st.popover("👎" if rating != "down" else "👎 ✓"):
        category = st.selectbox("What went wrong?", options=AGENT_FEEDBACK_CATEGORY_LABELS, key=f"agent_category_{run_id}")
        comment = st.text_area("Optional details", max_chars=500, key=f"agent_comment_{run_id}")
        st.button("Submit", key=f"agent_down_submit_{run_id}", disabled=category is None, on_click=_save_down)
with cols[2]:
    if rating is not None:
        st.button("Clear feedback", key=f"agent_clear_{run_id}", on_click=_clear)
```

Resolve the completed run by `(group_id, user_id)` in the service. Do not import Vanna training functions.

- [ ] **Step 5: Render once after the completed group**

At the end of the grouped branch in `render_message_group`, call the agent feedback renderer only when the service resolves a completed owned run for the group. Keep `_render_summary` unchanged for legacy Vanna feedback.

- [ ] **Step 6: Run focused and neighboring tests**

Run: `uv run pytest tests/views/test_agent_feedback.py tests/utils/test_message_grouping.py tests/utils/test_thumbs_down_feedback.py tests/views/test_admin_feedback.py tests/agent/test_runtime_agent_feedback.py -q`

Expected: all tests pass and existing Vanna feedback behavior remains unchanged.

- [ ] **Step 7: Commit**

```bash
git add views/agent_feedback.py agent/runtime.py utils/chat_bot_helper.py tests/views/test_agent_feedback.py tests/utils/test_message_grouping.py tests/agent/test_runtime_agent_feedback.py
git commit -m "feat(chat): add feedback to agentic answers"
```

---

### Task 4: Normalized Cases and Immutable Feedback Snapshots

**Files:**
- Create: `evals/cases.py`
- Create: `tests/evals/test_cases.py`
- Modify: `evals/matrix.py:19-50`

**Interfaces:**
- Produces: `NormalizedCase`, `NormalizedTurn`, `normalize_curated_conversation(planned: PlannedConversation) -> NormalizedCase`, `snapshot_feedback_case(feedback_id: int, admin_user_id: int) -> EvaluationCase`, and `promote_feedback_case(case_id: str, admin_user_id: int, definition: CuratedCaseDraft) -> EvaluationCase`.

- [ ] **Step 1: Write failing normalization and snapshot tests**

Use in-memory SQLite to verify a feedback snapshot contains the exact question, final answer, selected patient source ID, message history, tool evidence, feedback category/comment, source IDs, logging mode, and retention expiry. Assert changing the live feedback after snapshot creation does not alter `payload_json`.

Verify:

```python
assert execution_mode([feedback_case]) == "synchronous"
assert execution_mode([feedback_case, other_feedback_case]) == "asynchronous"
assert execution_mode([curated_case]) == "asynchronous"
```

Verify scrubbed evidence excludes full result rows. Verify disabled-mode feedback raises `SnapshotUnavailable("Exact replay is unavailable because agent logging is disabled.")`.

- [ ] **Step 2: Run case tests and verify they fail**

Run: `uv run pytest tests/evals/test_cases.py -q`

Expected: import failure for `evals.cases`.

- [ ] **Step 3: Implement source-neutral dataclasses**

```python
@dataclass(frozen=True)
class NormalizedTurn:
    role: Literal["main", "followup"]
    prompt: str

@dataclass(frozen=True)
class NormalizedCase:
    case_id: str
    version: int
    source_type: Literal["feedback", "curated"]
    title: str
    patient_source_id: str
    patient_label: str
    turns: tuple[NormalizedTurn, ...]
    message_history: tuple[dict, ...]
    original_answer: str | None
    original_evidence: tuple[dict, ...]
    reviewer_guidance: str

@dataclass(frozen=True)
class CuratedCaseDraft:
    prompt: str
    followups: tuple[str, ...]
    patient_requirements: str
    expected_behavior: str
    reviewer_guidance: str

class SnapshotUnavailable(RuntimeError):
    pass
```

Serialize with one explicit `to_payload()` and `from_payload()` pair; include `schema_version = 1`.

- [ ] **Step 4: Implement feedback snapshot validation**

Require admin role at the service boundary. Load `AgentRunFeedback`, `AgentRun`, originating `Message`, and `AgentRunEvent` rows in one session. Reject missing patient, question, final answer, or reconstructable history. Calculate `expires_at` from `AgentLoggingConfig.retention_days`; `0` means no expiry. Persist a new immutable `EvaluationCase` version before returning.

- [ ] **Step 5: Implement promotion without YAML mutation**

`promote_feedback_case` creates a new `source_type="curated"`, `status="draft"` case linked by `promoted_from_case_id`. Require non-empty reusable prompt, patient-selection requirement, expected behavior, and reviewer guidance. Do not open or write `evals/questions.yaml`.

- [ ] **Step 6: Run case tests**

Run: `uv run pytest tests/evals/test_cases.py tests/evals/test_matrix.py -q`

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add evals/cases.py evals/matrix.py tests/evals/test_cases.py
git commit -m "feat(evals): normalize curated and feedback cases"
```

---

### Task 5: Reusable Evaluation Executor and CLI Adapter

**Files:**
- Create: `evals/executor.py`
- Modify: `scripts/agent_eval_harness.py:69-160,214-251`
- Modify: `evals/judge.py:49-62`
- Create: `tests/evals/test_executor.py`
- Modify: `tests/evals/test_collect.py`
- Create: `tests/scripts/test_agent_eval_harness.py`

**Interfaces:**
- Consumes: `NormalizedCase` from Task 4.
- Produces: `EvaluationResources`, `CaseExecution`, `execute_case(case, resources) -> CaseExecution`, and `execute_cases(cases, resources, on_case_complete) -> list[CaseExecution]`.

- [ ] **Step 1: Write failing executor tests**

Use fake runner, patient resolver, adapter, and judge. Verify follow-up turns receive prior `all_messages`; tool evidence and latency survive normalization; judge receives the original feedback concern; judge failure yields `judge=None`; patient resolution failure returns a failed execution without replacing the patient; and the callback runs after every case.

- [ ] **Step 2: Run executor tests and verify they fail**

Run: `uv run pytest tests/evals/test_executor.py tests/scripts/test_agent_eval_harness.py -q`

Expected: import failure for `evals.executor`.

- [ ] **Step 3: Extract the reusable executor**

Define:

```python
@dataclass
class EvaluationResources:
    runner: AgenticRunner
    adapter: AnalyticsDbAdapter
    rag: Any
    judge: Any | None

@dataclass(frozen=True)
class CaseExecution:
    case_id: str
    status: Literal["completed", "failed"]
    patient: dict
    turns: tuple[dict, ...]
    error_type: str | None
    error_message: str | None

async def execute_case(case: NormalizedCase, resources: EvaluationResources) -> CaseExecution:
    # resolve exact source_id; clear SQL log; build deps; execute turns;
    # preserve history; judge each turn; return a plain serializable result.

async def execute_cases(
    cases: Sequence[NormalizedCase],
    resources: EvaluationResources,
    on_case_complete: Callable[[CaseExecution], None],
) -> list[CaseExecution]:
    results = []
    for case in cases:
        result = await execute_case(case, resources)
        on_case_complete(result)
        results.append(result)
    return results
```

Move `_build_deps`, resource construction, and `_run_conversation` behavior out of the script. Keep PHI out of exception logs; persist exception type plus a bounded user-facing message in results.

- [ ] **Step 4: Extend judge guidance for feedback cases**

Add an optional `reviewer_guidance` argument to `render_judge_prompt` and `judge_turn`. Render it under `USER FEEDBACK CONCERN:`. Keep the verdict values `looks_correct`, `looks_wrong`, and `unsure`; do not add a human pass/fail value.

- [ ] **Step 5: Adapt the CLI**

The CLI still parses questions/roster, builds the matrix, normalizes it, executes through `execute_cases`, writes JSON after each case, and emits the same `results:` and `next:` lines. Keep `--dry-run`, `--suggest-patients`, `--only`, `--skip-judge`, `--limit-patients`, and `--out` behavior.

- [ ] **Step 6: Run executor and CLI tests**

Run: `uv run pytest tests/evals/test_executor.py tests/evals/test_collect.py tests/scripts/test_agent_eval_harness.py -q`

Expected: all tests pass.

- [ ] **Step 7: Run a real dry run**

Run: `uv run python scripts/agent_eval_harness.py --dry-run --limit-patients 1`

Expected: exit 0 and print a non-zero conversation and turn count without calling an LLM or warehouse.

- [ ] **Step 8: Commit**

```bash
git add evals/executor.py evals/judge.py scripts/agent_eval_harness.py tests/evals/test_executor.py tests/evals/test_collect.py tests/scripts/test_agent_eval_harness.py
git commit -m "refactor(evals): share execution with the CLI"
```

---

### Task 6: Durable Run Service, Worker, Recovery, and Retention

**Files:**
- Extend: `orm/evaluation_functions.py`
- Create: `evals/worker.py`
- Modify: `app.py:23-44`
- Create: `tests/evals/test_run_service.py`
- Create: `tests/evals/test_worker.py`
- Create: `tests/evals/test_retention.py`

**Interfaces:**
- Produces: `create_evaluation_run(case_ids: list[int], admin_user_id: int) -> EvaluationRunView`, `execute_synchronous_run(run_id: str, admin_user_id: int) -> EvaluationRunView`, `record_final_verdict(result_id: int, admin_user_id: int, verdict: Literal['correct','incorrect','cant_tell'], note: str | None) -> None`, `EvaluationQueue` protocol, `SqliteEvaluationQueue`, and `start_evaluation_worker() -> None`.

- [ ] **Step 1: Write failing run lifecycle tests**

Verify one feedback case creates `single_feedback/synchronous`; two feedback cases create `feedback_batch/asynchronous`; one curated case creates `curated_subset/asynchronous`; all active curated cases create `full_suite/asynchronous`. Assert a non-admin receives `PermissionError` before any row is inserted.

Verify case results are inserted atomically with the run, case completions commit independently, and statuses aggregate to `completed`, `completed_with_errors`, or `failed`.

- [ ] **Step 2: Write failing queue and recovery tests**

With a file-backed temporary SQLite DB, start two queue claimers concurrently and assert only one claims a queued run. Assert a second queued run stays queued while one is active. Expire a heartbeat, call `mark_interrupted_runs`, and assert only pending/failed cases are eligible for resume. Assert cancellation stops after the current case and marks remaining pending results cancelled.

- [ ] **Step 3: Run service and worker tests and verify they fail**

Run: `uv run pytest tests/evals/test_run_service.py tests/evals/test_worker.py tests/evals/test_retention.py -q`

Expected: missing run-service and worker interfaces.

- [ ] **Step 4: Implement admin-checked run creation and review**

Use `RoleTypeEnum.ADMIN` from the database, not session-state trust. `create_evaluation_run` snapshots all selected source cases first, then inserts the run and ordered result rows in one transaction. `record_final_verdict` accepts only the three final values, appends `EvaluationReviewEvent`, and updates the result in one transaction.

Expose service results through this immutable view rather than leaking attached SQLAlchemy rows outside the service transaction:

```python
@dataclass(frozen=True)
class EvaluationRunView:
    run_id: str
    run_type: str
    execution_mode: Literal["synchronous", "asynchronous"]
    status: str
    total_cases: int
    completed_cases: int
    failed_cases: int
```

- [ ] **Step 5: Implement the replaceable queue contract**

```python
class EvaluationQueue(Protocol):
    def claim_next(self, worker_id: str) -> str | None: ...
    def heartbeat(self, run_id: str, worker_id: str) -> None: ...
    def finish(self, run_id: str, status: str) -> None: ...

class SqliteEvaluationQueue:
    def claim_next(self, worker_id: str) -> str | None:
        # BEGIN IMMEDIATE; refuse claim if any run is running;
        # claim oldest queued row; commit; return public run_id.
```

Execute cases sequentially with `evals.executor`. Commit each result before claiming the next case. Update heartbeat before and after each case. On completion, create one PHI-free `AdminNotification` for the requesting admin.

- [ ] **Step 6: Start one process-local worker after database bootstrap**

`start_evaluation_worker` uses a module lock and cached daemon thread so Streamlit reruns cannot create duplicate workers. The thread polls every five seconds, marks heartbeats older than two minutes interrupted, and processes one run at a time. `app.py` starts it only after `_bootstrap_db()` succeeds. The database claim remains authoritative if more than one app process exists.

- [ ] **Step 7: Apply retention**

`purge_expired_evaluation_payloads(now)` clears case `payload_json` and result `result_json` for expired cases, sets case status `expired`, and retains only run IDs, statuses, timestamps, counts, and the fact that details expired. It must not delete review audit records. Call it once on worker start and once every 24 hours.

- [ ] **Step 8: Run lifecycle tests**

Run: `uv run pytest tests/evals/test_run_service.py tests/evals/test_worker.py tests/evals/test_retention.py tests/orm/test_engine_multithread.py -q`

Expected: all tests pass.

- [ ] **Step 9: Commit**

```bash
git add orm/evaluation_functions.py evals/worker.py app.py tests/evals/test_run_service.py tests/evals/test_worker.py tests/evals/test_retention.py
git commit -m "feat(evals): execute durable queued runs"
```

---

### Task 7: Admin Evaluation Catalog and Launch UI

**Files:**
- Create: `views/admin_evaluations.py`
- Modify: `views/admin.py:18-65`
- Create: `tests/views/test_admin_evaluations.py`
- Create: `tests/views/test_admin.py`

**Interfaces:**
- Consumes: Task 4 snapshot functions and Task 6 run service.
- Produces: Admin source catalog, selection summary, synchronous launch, asynchronous launch, queue status, and cancellation UI.

- [ ] **Step 1: Write failing authorization and launch tests**

Mock Streamlit and service calls. Assert `_guard_admin` stops non-admins. Assert the feedback source exposes user, organization, patient, date, category, and question filters. Assert curated source supports selected cases/patients and full-suite selection. Assert the launch summary displays case, conversation, turn counts, and execution mode.

Test launch dispatch:

```python
launch_selected([feedback_case_id])
mock_execute_sync.assert_called_once()

launch_selected([feedback_case_id, second_feedback_case_id])
mock_execute_sync.assert_not_called()
assert mock_create_run.return_value.execution_mode == "asynchronous"

launch_selected([curated_case_id])
mock_execute_sync.assert_not_called()
```

- [ ] **Step 2: Run Admin UI tests and verify they fail**

Run: `uv run pytest tests/views/test_admin_evaluations.py tests/views/test_admin.py -q`

Expected: import failure for `views.admin_evaluations`.

- [ ] **Step 3: Add the Evaluations Admin tab**

Import `admin_evaluations`, change tabs to `Users · Training · Analytics · Audit · Feedback · Evaluations`, and call `admin_evaluations.render(days_int)` in the sixth tab. Preserve the shared time-range control.

- [ ] **Step 4: Build the launch catalog**

Use `st.segmented_control` for `Thumbs-down feedback` and `Curated suite`; `st.dataframe(..., on_select="rerun", selection_mode="multi-row")` for case selection; and a bordered summary showing counts, execution mode, model/warehouse warning, and active-run queue state. Keep PHI inside the Admin page.

- [ ] **Step 5: Implement launch behavior**

For one feedback case, show `st.status` while `execute_synchronous_run` runs, then set `st.session_state["evaluation_run_id"]` and rerun into the report. For every asynchronous launch, show a PHI-free toast with the run ID and queue state. Catch `SnapshotUnavailable` and identify the missing context without exposing another user's content.

- [ ] **Step 6: Run Admin launch tests**

Run: `uv run pytest tests/views/test_admin_evaluations.py tests/views/test_admin.py tests/views/test_admin_feedback.py -q`

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add views/admin_evaluations.py views/admin.py tests/views/test_admin_evaluations.py tests/views/test_admin.py
git commit -m "feat(admin): launch authenticated evaluations"
```

---

### Task 8: Authenticated Results, Human Verdicts, Promotion, and Notifications

**Files:**
- Extend: `views/admin_evaluations.py`
- Extend: `orm/evaluation_functions.py`
- Extend: `tests/views/test_admin_evaluations.py`
- Create: `tests/evals/test_review_workflow.py`
- Create: `tests/evals/test_notifications.py`

**Interfaces:**
- Consumes: durable results and review operations from Task 6.
- Produces: authenticated original-versus-rerun report, judge triage, final verdict history, promotion dialog, and persistent notification actions.

- [ ] **Step 1: Write failing report authorization tests**

Assert `get_evaluation_run_report(run_id, actor_user_id)` rejects a doctor and returns data to an admin. Assert returned notifications expose only notification ID, run ID, kind, status, timestamps, and counts; patient/question/answer fields must be absent.

- [ ] **Step 2: Write failing review and promotion tests**

Verify a judge verdict cannot populate `final_verdict`. Verify each final-verdict change appends an event with old/new values and reviewer. Verify promotion creates a draft curated case and no test opens `evals/questions.yaml` for writing.

- [ ] **Step 3: Run report workflow tests and verify they fail**

Run: `uv run pytest tests/evals/test_review_workflow.py tests/evals/test_notifications.py tests/views/test_admin_evaluations.py -q`

Expected: missing report, notification, or review behavior.

- [ ] **Step 4: Render the authenticated report**

Render run status/counts first. For feedback cases, show the concern, then two columns for original and rerun answers. Under each answer, render tool summaries, SQL, reliability, and timing only when present under the logging mode. Render LLM triage separately from a required admin verdict selector with `correct`, `incorrect`, and `Can't tell`.

For curated cases, render the existing scorecard concepts from `evals/report.py` using Streamlit components; do not embed or link a generated HTML file.

- [ ] **Step 5: Add promotion and resume/cancel actions**

The promotion dialog collects reusable prompt/follow-ups, patient requirements, expected behavior, and reviewer guidance; save as draft through Task 4. Interrupted runs expose `Resume unfinished`; queued/running runs expose cancellation according to Task 6 rules.

- [ ] **Step 6: Add persistent notifications**

At the top of the Evaluations tab, list unread notifications with `View report` and `Mark read`. `View report` writes only the run ID to session state and opens the authenticated report renderer. The notification text is `Evaluation <run_id> completed: <completed>/<total> cases.`

- [ ] **Step 7: Run report workflow tests**

Run: `uv run pytest tests/evals/test_review_workflow.py tests/evals/test_notifications.py tests/views/test_admin_evaluations.py -q`

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add views/admin_evaluations.py orm/evaluation_functions.py tests/views/test_admin_evaluations.py tests/evals/test_review_workflow.py tests/evals/test_notifications.py
git commit -m "feat(admin): review evaluation reports"
```

---

### Task 9: Documentation and Full Verification

**Files:**
- Modify: `evals/README.md`
- Create: `tests/test_app_evaluation_worker.py`

**Interfaces:**
- Produces: operator instructions, user workflow documentation, example worker settings, and final regression evidence.

- [ ] **Step 1: Document exact behavior**

Update `evals/README.md` with:

- agentic feedback versus Vanna training semantics;
- synchronous and asynchronous selection rules;
- durable statuses and recovery;
- process-local worker startup and the one-active-run limit;
- logging-mode and retention effects;
- developer CLI dry-run, live run, JSON, and HTML commands;
- the statement that production reports remain authenticated application pages.

Document the agentic thumbs controls and Admin Evaluations workflow in `evals/README.md`. Add this example configuration there rather than creating a secrets file that could be mistaken for deployable credentials:

```toml
[agent_evaluations]
worker_enabled = true
poll_interval_s = 5
heartbeat_timeout_s = 120
```

- [ ] **Step 2: Add app startup coverage**

In `tests/test_app_evaluation_worker.py`, execute `app.py` with Streamlit, authentication, and page dependencies patched; patch `start_evaluation_worker`; assert it runs once after successful DB bootstrap and never runs when migration bootstrap returns an error.

- [ ] **Step 3: Run focused evaluation and feedback tests**

Run:

```bash
uv run pytest \
  tests/orm/test_evaluation_models.py \
  tests/orm/test_authenticated_evaluation_migration.py \
  tests/orm/test_agent_run_feedback_functions.py \
  tests/views/test_agent_feedback.py \
  tests/evals \
  tests/views/test_admin_evaluations.py \
  tests/test_app_evaluation_worker.py -q
```

Expected: all tests pass.

- [ ] **Step 4: Run legacy feedback and agent regressions**

Run:

```bash
uv run pytest \
  tests/utils/test_thumbs_down_feedback.py \
  tests/views/test_admin_feedback.py \
  tests/utils/test_message_grouping.py \
  tests/agent -q
```

Expected: all tests pass; Vanna feedback/training and agent rendering remain intact.

- [ ] **Step 5: Run migration, lint, and full test suite**

Run:

```bash
uv run alembic upgrade head
uv run alembic current
uv run ruff check
uv run pytest
```

Expected: Alembic reports `d91c7a4ef201 (head)`, Ruff exits 0, and pytest exits 0.

- [ ] **Step 6: Run the CLI compatibility smoke test**

Run: `uv run python scripts/agent_eval_harness.py --dry-run --limit-patients 1`

Expected: exit 0 with resolved conversations and turns and no LLM/warehouse execution.

- [ ] **Step 7: Commit**

```bash
git add evals/README.md tests/test_app_evaluation_worker.py
git commit -m "docs(evals): document authenticated evaluation workflow"
```

- [ ] **Step 8: Inspect the final branch**

Run:

```bash
git status --short
git diff origin/main...HEAD --check
git log --oneline origin/main..HEAD
```

Expected: clean status, no whitespace errors, and one reviewed commit per task.