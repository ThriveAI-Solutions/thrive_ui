import enum
import json
import os
from decimal import Decimal
from pathlib import Path

import pandas as pd
import streamlit as st
from sqlalchemy import (
    TIMESTAMP,
    Boolean,
    Column,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    create_engine,
    func,
)
from sqlalchemy import Enum as SqlEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

from utils.enums import MessageType, RoleType, ThemeType
from utils.quick_logger import get_logger

logger = get_logger(__name__)

_DEFAULT_SQLITE_PATH = "./pgDatabase/db.sqlite3"


def _get_database_url() -> str:
    """Resolve the SQLite URL.

    Priority: THRIVE_SQLITE_PATH env var → `[sqlite].database` from Streamlit
    secrets → hardcoded default. The env var takes priority so tests and the
    alembic CLI can override the path even when secrets.toml is present.
    """
    override = os.environ.get("THRIVE_SQLITE_PATH")
    if override:
        return f"sqlite:///{override}"
    # Narrow except: only the "no secrets file / no Streamlit runtime" case
    # falls back to default. A TOML parse error or other surprise propagates
    # so the dev sees it instead of silently booting against the wrong DB.
    try:
        db_settings = st.secrets.get("sqlite", {"database": _DEFAULT_SQLITE_PATH})
    except (FileNotFoundError, AttributeError) as exc:
        logger.debug("st.secrets unavailable (%s); using default DB path", exc)
        db_settings = {"database": _DEFAULT_SQLITE_PATH}
    return f"sqlite:///{db_settings['database']}"


DATABASE_URL = _get_database_url()
# The agentic flow (agent/runtime.py) writes this DB from two threads at once:
# the agent's audit rows on a dedicated asyncio loop thread, and Message.save on
# the Streamlit script thread. They can hold overlapping write transactions, so
# SQLite's busy-wait is functionally relied upon here. SQLAlchemy's pysqlite
# dialect already allows cross-thread pooled connections and the sqlite3 busy
# timeout already defaults to 5s, so the original engine was adequate; we set
# these explicitly to record that cross-thread use is intended and to widen the
# write-lock wait to 30s for headroom under contention.
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False, "timeout": 30},
)

# Create a configured "Session" class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create a Base class for the models to inherit from
Base = declarative_base()


def content_to_json(type, content):
    if type == MessageType.DATAFRAME.value and not isinstance(content, str):
        panda_frame = pd.DataFrame(content)
        return panda_frame.to_json(date_format="iso")
    if type == MessageType.PLOTLY_CHART.value and not isinstance(content, str):
        json_content = content.to_json()
        return json_content
    if type == MessageType.FOLLOWUP.value and isinstance(content, list):
        return json.dumps(content)
    return content


class RoleTypeEnum(enum.IntEnum):
    ADMIN = 0
    DOCTOR = 1
    NURSE = 2
    PATIENT = 3


# ============== Logging Enums ==============


class ActivityType(enum.Enum):
    """Types of user activity events for audit logging."""

    LOGIN = "login"
    LOGOUT = "logout"
    LOGIN_FAILED = "login_failed"
    SETTINGS_CHANGE = "settings_change"
    PASSWORD_CHANGE = "password_change"
    THEME_CHANGE = "theme_change"
    LLM_PREFERENCE_CHANGE = "llm_preference_change"
    CLEAR_MESSAGES = "clear_messages"


class AdminActionType(enum.Enum):
    """Types of admin actions for audit logging."""

    USER_CREATE = "user_create"
    USER_UPDATE = "user_update"
    USER_DELETE = "user_delete"
    USER_PASSWORD_RESET = "user_password_reset"
    ROLE_CHANGE = "role_change"
    TRAINING_APPROVE = "training_approve"
    TRAINING_REJECT = "training_reject"
    TRAINING_BULK_APPROVE = "training_bulk_approve"
    TRAINING_BULK_REJECT = "training_bulk_reject"
    TRAINING_ADD = "training_add"
    TRAINING_DELETE = "training_delete"
    TRAINING_IMPORT = "training_import"
    USER_IMPORT = "user_import"
    USER_EXPORT = "user_export"


class ErrorCategory(enum.Enum):
    """Categories of errors for structured error logging."""

    SQL_GENERATION = "sql_generation"
    SQL_EXECUTION = "sql_execution"
    SQL_VALIDATION = "sql_validation"
    CHART_GENERATION = "chart_generation"
    SUMMARY_GENERATION = "summary_generation"
    RAG_RETRIEVAL = "rag_retrieval"
    LLM_API = "llm_api"
    DATABASE_CONNECTION = "database_connection"
    AUTHENTICATION = "authentication"
    TRAINING = "training"
    GENERAL = "general"


class ErrorSeverity(enum.Enum):
    """Severity levels for error logging."""

    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class UserRole(Base):
    __tablename__ = "thrive_user_role"
    id = Column(Integer, primary_key=True)
    role_name = Column(String(50), nullable=False, unique=True)
    description = Column(String)
    role = Column(SqlEnum(RoleTypeEnum), nullable=False)
    users = relationship("User", back_populates="role")


class User(Base):
    __tablename__ = "thrive_user"
    id = Column(Integer, primary_key=True)
    # Required per Epic #179 — every user must resolve to a UserRole so the
    # audit-trail / role-restricted RAG paths never see NULL role rows. The
    # Alembic revision require_user_email_org_role (rev 7b3a1f0c92d4) backfills
    # existing NULL rows with the PATIENT role before applying NOT NULL.
    user_role_id = Column(Integer, ForeignKey("thrive_user_role.id"), nullable=False)
    # 320 aligns with email max length for OIDC JIT users (username defaults to email or sub).
    username = Column(String(320), nullable=False, unique=True)
    first_name = Column(String(50), nullable=False)
    last_name = Column(String(50), nullable=False)
    # OIDC-only users store a reserved non-hash sentinel here so legacy SQLite
    # databases with NOT NULL password constraints can JIT-provision them.
    password = Column(String(255), nullable=False)
    # OIDC fields. `okta_sub` is NULL for local-only users; populated for users
    # who authenticate via Okta SSO. See
    # docs/superpowers/specs/2026-05-01-okta-oidc-integration-design.md §5.
    okta_sub = Column(String(255), nullable=True, unique=True)
    # `email` and `organization` are required per Epic #179 — the audit trail
    # organization filter and role-restricted RAG retrieval both depend on them
    # being populated. OIDC JIT provisioning enforces a hard error if the IdP
    # omits `email`, and derives `organization` from the email domain when the
    # claim is missing (logged at WARN). See utils/okta_auth.sync_okta_user_to_db.
    email = Column(String(320, collation="NOCASE"), nullable=False, unique=True)
    organization = Column(String(120), nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())
    show_sql = Column(Boolean, default=True)
    show_table = Column(Boolean, default=True)
    show_plotly_code = Column(Boolean, default=False)
    show_chart = Column(Boolean, default=False)
    show_question_history = Column(Boolean, default=True)
    show_summary = Column(Boolean, default=True)
    voice_input = Column(Boolean, default=False)
    speak_summary = Column(Boolean, default=False)
    show_suggested = Column(Boolean, default=False)
    show_followup = Column(Boolean, default=False)
    show_elapsed_time = Column(Boolean, default=True)
    llm_fallback = Column(Boolean, default=False)
    confirm_magic_commands = Column(Boolean, default=True)  # True = show popup, False = auto-execute
    show_community_engagement = Column(Boolean, default=False)
    agentic_mode = Column(Boolean, default=True)
    min_message_id = Column(Integer, default=0)
    theme = Column(String(50), default=ThemeType.THRIVEAI.value)
    selected_llm_provider = Column(String(50), default=None)
    selected_llm_model = Column(String(100), default=None)

    role = relationship("UserRole", back_populates="users")
    messages = relationship(
        "Message", back_populates="user", cascade="all, delete-orphan", foreign_keys="[Message.user_id]"
    )

    def to_dict(self):
        return {
            "id": self.id,
            "user_role_id": self.user_role_id,
            "username": self.username,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "show_sql": self.show_sql,
            "show_table": self.show_table,
            "show_plotly_code": self.show_plotly_code,
            "show_chart": self.show_chart,
            "show_question_history": self.show_question_history,
            "show_summary": self.show_summary,
            "voice_input": self.voice_input,
            "speak_summary": self.speak_summary,
            "show_suggested": self.show_suggested,
            "show_followup": self.show_followup,
            "show_elapsed_time": self.show_elapsed_time,
            "llm_fallback": self.llm_fallback,
            "confirm_magic_commands": self.confirm_magic_commands,
            "show_community_engagement": self.show_community_engagement,
            "min_message_id": self.min_message_id,
            "theme": self.theme,
            "selected_llm_provider": self.selected_llm_provider,
            "selected_llm_model": self.selected_llm_model,
            "agentic_mode": self.agentic_mode,
        }


class Message(Base):
    __tablename__ = "thrive_message"
    __table_args__ = (
        Index("ix_thrive_message_user_id", "user_id"),
        Index("ix_thrive_message_created_at", "created_at"),
        Index("ix_thrive_message_type", "type"),
        Index("ix_thrive_message_feedback", "feedback"),
        Index("ix_thrive_message_training_status", "training_status"),
    )
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("thrive_user.id"))
    group_id = Column(String(50))  # UUID to group messages in the same flow
    role = Column(String(50), nullable=False)
    content = Column(String, nullable=False)
    type = Column(String(50), nullable=False)
    feedback = Column(String(50))
    feedback_comment = Column(String(500))  # Optional user feedback comment when thumbs down
    training_status = Column(String(20))  # 'pending', 'approved', 'rejected', or None (auto-approved for admin)
    reviewed_by = Column(Integer, ForeignKey("thrive_user.id"))  # Admin who reviewed
    reviewed_at = Column(TIMESTAMP)  # When the review happened
    query = Column(String)
    question = Column(String(1000))
    dataframe = Column(String)
    elapsed_time = Column(Numeric(10, 6))
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())

    # ORM relationship back to user
    user = relationship("User", back_populates="messages", foreign_keys=[user_id])
    # ORM relationship for the reviewer (admin who approved/rejected)
    reviewer = relationship("User", foreign_keys=[reviewed_by])

    def __init__(
        self,
        role: RoleType,
        content: str,
        type: MessageType,
        query: str = None,
        question: str = None,
        dataframe: pd.DataFrame | str = None,
        elapsed_time: Decimal = None,
        user_id: int = None,
        group_id: str = None,
    ):
        # Try to get user_id from session_state if not provided directly
        if user_id is None:
            try:
                # Get user_id from session_state cookies if available
                if hasattr(st.session_state, "cookies") and st.session_state.cookies is not None:
                    user_id_str = st.session_state.cookies.get("user_id")
                    if user_id_str:
                        user_id = json.loads(user_id_str)

                # If still None, default to 1 for testing purposes
                if user_id is None:
                    user_id = 1
            except Exception:
                # Default to user_id 1 for testing purposes if there's any error
                user_id = 1

        self.user_id = user_id
        self.group_id = group_id
        self.role = role.value
        self.content = content_to_json(type.value, content)
        self.type = type.value
        self.feedback = None
        self.query = query
        self.question = question
        self.elapsed_time = elapsed_time

        # Serialize the DataFrame to JSON if provided
        if dataframe is not None:
            # Handle both DataFrame objects and serialized JSON strings
            if isinstance(dataframe, pd.DataFrame):
                self.dataframe = dataframe.to_json(date_format="iso")  # Convert DataFrame to JSON
            elif isinstance(dataframe, str):
                # Assume it's already a JSON string
                self.dataframe = dataframe
            else:
                # Try to convert other types to string
                self.dataframe = str(dataframe)
        else:
            self.dataframe = None

    def to_dict(self):
        return {"role": self.role, "content": self.content, "question": self.question}

    def save(self):
        session = SessionLocal()

        self.content = content_to_json(self.type, self.content)

        # Add the new message to the session and commit
        if self.id is None:
            session.add(self)
            session.commit()
            # Refresh the db_message object to get the auto-generated fields
            session.refresh(self)
        else:
            session.merge(self)
            session.commit()

        # Close the session
        session.close()

        return self


# ============== Logging Models ==============


class LLMContext(Base):
    """Captures the full RAG context sent to LLM for each SQL generation."""

    __tablename__ = "thrive_llm_context"
    __table_args__ = (
        Index("ix_thrive_llm_context_message_id", "message_id"),
        Index("ix_thrive_llm_context_user_id", "user_id"),
        Index("ix_thrive_llm_context_created_at", "created_at"),
        Index("ix_thrive_llm_context_llm_provider", "llm_provider"),
    )

    id = Column(Integer, primary_key=True)
    message_id = Column(Integer, ForeignKey("thrive_message.id"), nullable=True)
    group_id = Column(String(50))  # Same UUID from Message for correlation
    user_id = Column(Integer, ForeignKey("thrive_user.id"), nullable=True)

    # LLM Configuration
    llm_provider = Column(String(50))  # anthropic, ollama, gemini, openai
    llm_model = Column(String(100))  # claude-3-5-sonnet, llama3.2:latest, etc.

    # RAG Context Retrieved
    ddl_statements = Column(String)  # JSON array of DDL strings sent to LLM
    documentation_snippets = Column(String)  # JSON array of doc strings sent to LLM
    question_sql_examples = Column(String)  # JSON array of {question, sql} pairs
    ddl_count = Column(Integer, default=0)
    doc_count = Column(Integer, default=0)
    example_count = Column(Integer, default=0)

    # The full prompt sent to LLM (optional, for debugging)
    full_prompt = Column(String)  # JSON serialized message list

    # Token Usage (if available from provider)
    input_tokens = Column(Integer)
    output_tokens = Column(Integer)
    total_tokens = Column(Integer)

    # Timing
    retrieval_time_ms = Column(Integer)  # Time to retrieve RAG context
    inference_time_ms = Column(Integer)  # Time for LLM to respond
    total_time_ms = Column(Integer)  # Total end-to-end time

    # Generated Output
    question = Column(String(1000))  # The user question
    generated_sql = Column(String)  # The SQL that was generated
    thinking_content = Column(String)  # For thinking models, the reasoning trace

    created_at = Column(TIMESTAMP, server_default=func.now())

    # Relationships
    message = relationship("Message", foreign_keys=[message_id])
    user = relationship("User", foreign_keys=[user_id])


class UserActivity(Base):
    """Tracks user activity for audit logging."""

    __tablename__ = "thrive_user_activity"
    __table_args__ = (
        Index("ix_thrive_user_activity_user_id", "user_id"),
        Index("ix_thrive_user_activity_activity_type", "activity_type"),
        Index("ix_thrive_user_activity_created_at", "created_at"),
    )

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("thrive_user.id"), nullable=True)  # Nullable for failed logins
    username = Column(String(50))  # Store username even if user lookup fails
    activity_type = Column(String(50), nullable=False)  # From ActivityType enum

    # Activity Details
    description = Column(String(500))  # Human-readable description
    old_value = Column(String)  # JSON: previous state for settings changes
    new_value = Column(String)  # JSON: new state for settings changes

    # Request Context
    ip_address = Column(String(45))  # IPv4 or IPv6
    user_agent = Column(String(500))

    created_at = Column(TIMESTAMP, server_default=func.now())

    # Relationships
    user = relationship("User", foreign_keys=[user_id])


class AdminAction(Base):
    """Tracks admin actions for compliance and audit logging."""

    __tablename__ = "thrive_admin_action"
    __table_args__ = (
        Index("ix_thrive_admin_action_admin_id", "admin_id"),
        Index("ix_thrive_admin_action_action_type", "action_type"),
        Index("ix_thrive_admin_action_target_user_id", "target_user_id"),
        Index("ix_thrive_admin_action_created_at", "created_at"),
    )

    id = Column(Integer, primary_key=True)
    admin_id = Column(Integer, ForeignKey("thrive_user.id"), nullable=False)
    action_type = Column(String(50), nullable=False)  # From AdminActionType enum

    # Target Information
    target_user_id = Column(Integer, ForeignKey("thrive_user.id"), nullable=True)
    target_username = Column(String(50))  # Denormalized for deleted users
    target_entity_type = Column(String(50))  # 'user', 'training_data', 'message', etc.
    target_entity_id = Column(String(100))  # ID of affected entity

    # Action Details
    description = Column(String(500))
    old_value = Column(String)  # JSON: previous state
    new_value = Column(String)  # JSON: new state
    affected_count = Column(Integer)  # For bulk operations

    # Success/Failure
    success = Column(Boolean, default=True)
    error_message = Column(String(500))

    created_at = Column(TIMESTAMP, server_default=func.now())

    # Relationships
    admin = relationship("User", foreign_keys=[admin_id])
    target_user = relationship("User", foreign_keys=[target_user_id])


class ErrorLog(Base):
    """Structured error logging for debugging and monitoring."""

    __tablename__ = "thrive_error_log"
    __table_args__ = (
        Index("ix_thrive_error_log_user_id", "user_id"),
        Index("ix_thrive_error_log_category", "category"),
        Index("ix_thrive_error_log_severity", "severity"),
        Index("ix_thrive_error_log_created_at", "created_at"),
        Index("ix_thrive_error_log_message_id", "message_id"),
    )

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("thrive_user.id"), nullable=True)
    message_id = Column(Integer, ForeignKey("thrive_message.id"), nullable=True)
    group_id = Column(String(50))  # Correlation with message flow

    # Error Classification
    category = Column(String(50), nullable=False)  # From ErrorCategory enum
    severity = Column(String(20), nullable=False)  # From ErrorSeverity enum

    # Error Details
    error_type = Column(String(100))  # Exception class name
    error_message = Column(String(2000))
    stack_trace = Column(String)  # Full traceback

    # Context
    question = Column(String(1000))  # User question that triggered error
    generated_sql = Column(String)  # SQL that failed (if applicable)
    llm_provider = Column(String(50))
    llm_model = Column(String(100))

    # Additional Context (JSON)
    context_data = Column(String)  # JSON with any additional debugging info

    # Resolution
    auto_retry_attempted = Column(Boolean, default=False)
    retry_successful = Column(Boolean)
    retry_count = Column(Integer, default=0)

    created_at = Column(TIMESTAMP, server_default=func.now())

    # Relationships
    user = relationship("User", foreign_keys=[user_id])
    message = relationship("Message", foreign_keys=[message_id])


class ToolCall(Base):
    __tablename__ = "thrive_tool_call"
    __table_args__ = (
        Index("ix_thrive_tool_call_session_id", "session_id"),
        Index("ix_thrive_tool_call_user_id", "user_id"),
        Index("ix_thrive_tool_call_created_at", "created_at"),
        Index("ix_thrive_tool_call_tool_name", "tool_name"),
        Index("ix_thrive_tool_call_selected_patient", "selected_patient_source_id"),
        Index("ix_thrive_tool_call_run_id", "run_id"),
        Index("ix_thrive_tool_call_run_call", "run_id", "call_index"),
        Index("ix_thrive_tool_call_tool_call_id", "tool_call_id"),
    )
    id = Column(Integer, primary_key=True)
    session_id = Column(String(64), nullable=False)
    user_id = Column(Integer, ForeignKey("thrive_user.id"), nullable=False)
    user_role = Column(Integer, nullable=False)
    selected_patient_source_id = Column(String(50), nullable=True)
    tool_name = Column(String(64), nullable=False)
    arguments_json = Column(Text, nullable=False)
    result_summary = Column(Text, nullable=False)
    elapsed_ms = Column(Integer, nullable=False)
    success = Column(Boolean, nullable=False)
    error = Column(Text, nullable=True)
    # --- Agentic run logging enrichment (2026-05 design) ---
    run_id = Column(String(36), nullable=True)
    tool_call_id = Column(String(100), nullable=True)
    call_index = Column(Integer, nullable=True)
    turn_index = Column(Integer, nullable=True)
    attempt_index = Column(Integer, nullable=False, default=1)
    started_event_seq = Column(Integer, nullable=True)
    completed_event_seq = Column(Integer, nullable=True)
    result_json = Column(Text, nullable=True)
    result_truncated = Column(Boolean, nullable=False, default=False)
    result_bytes = Column(Integer, nullable=True)
    result_hash = Column(String(64), nullable=True)
    sql_executed_json = Column(Text, nullable=True)
    sql_executed_truncated = Column(Boolean, nullable=False, default=False)
    sql_executed_bytes = Column(Integer, nullable=True)
    sql_executed_hash = Column(String(64), nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now(), nullable=False)


class AgentRun(Base):
    """One row per agentic question/turn. Query-friendly rollup over the
    append-only AgentRunEvent timeline."""

    __tablename__ = "thrive_agent_run"
    __table_args__ = (
        Index("ix_thrive_agent_run_run_id", "run_id", unique=True),
        Index("ix_thrive_agent_run_session_id", "session_id"),
        Index("ix_thrive_agent_run_user_id", "user_id"),
        Index("ix_thrive_agent_run_created_at", "created_at"),
        Index("ix_thrive_agent_run_group_id", "group_id"),
        Index("ix_thrive_agent_run_selected_patient", "selected_patient_source_id"),
        Index("ix_thrive_agent_run_status", "status"),
        Index("ix_thrive_agent_run_review_status", "review_status"),
        # Backs the LEFT JOIN added by the Question Audit Scope filter
        # (Epic #166 / Feature #167). Matching migration: 188ab391e291.
        Index("ix_thrive_agent_run_user_message_id", "user_message_id"),
    )

    id = Column(Integer, primary_key=True)
    run_id = Column(String(36), nullable=False)
    session_id = Column(String(64), nullable=False)
    group_id = Column(String(50), nullable=True)
    parent_run_id = Column(String(36), nullable=True)
    resume_reason = Column(String(40), nullable=True)
    user_message_id = Column(Integer, nullable=True)
    final_message_id = Column(Integer, nullable=True)
    user_id = Column(Integer, ForeignKey("thrive_user.id"), nullable=False)
    user_role = Column(Integer, nullable=False)
    question = Column(Text, nullable=True)

    selected_patient_source_id = Column(String(50), nullable=True)
    selected_patient_display_name = Column(String(255), nullable=True)
    selected_patient_dob = Column(String(50), nullable=True)
    selected_patient_selection_origin = Column(String(32), nullable=True)

    llm_provider = Column(String(50), nullable=True)
    llm_model = Column(String(100), nullable=True)
    model_settings_json = Column(Text, nullable=True)
    system_prompt_hash = Column(String(64), nullable=True)
    tool_schema_hash = Column(String(64), nullable=True)
    message_history_json = Column(Text, nullable=True)

    final_answer_text = Column(Text, nullable=True)
    input_tokens = Column(Integer, nullable=True)
    output_tokens = Column(Integer, nullable=True)
    total_tokens = Column(Integer, nullable=True)
    tool_call_count = Column(Integer, nullable=False, default=0)
    event_count = Column(Integer, nullable=False, default=0)
    total_elapsed_ms = Column(Integer, nullable=True)
    cap_reached = Column(String(20), nullable=True)
    status = Column(String(20), nullable=False, default="open")
    success = Column(Boolean, nullable=False, default=False)
    error_type = Column(String(100), nullable=True)
    error = Column(Text, nullable=True)
    stack_trace = Column(Text, nullable=True)

    review_status = Column(String(20), nullable=False, default="unreviewed")
    reviewed_by = Column(Integer, nullable=True)
    reviewed_at = Column(TIMESTAMP, nullable=True)
    review_notes = Column(Text, nullable=True)
    issue_url = Column(Text, nullable=True)

    logging_mode = Column(String(20), nullable=False, default="full")
    schema_version = Column(Integer, nullable=False, default=1)
    app_git_sha = Column(String(40), nullable=True)
    environment = Column(String(50), nullable=True)

    created_at = Column(TIMESTAMP, server_default=func.now(), nullable=False)
    completed_at = Column(TIMESTAMP, nullable=True)


class AgentRunEvent(Base):
    """Append-only ordered timeline; the source of truth for the inspector."""

    __tablename__ = "thrive_agent_run_event"
    __table_args__ = (
        Index("ix_thrive_agent_run_event_run_seq", "run_id", "seq", unique=True),
        Index("ix_thrive_agent_run_event_run_id", "run_id"),
        Index("ix_thrive_agent_run_event_type", "event_type"),
        Index("ix_thrive_agent_run_event_tool_name", "tool_name"),
        Index("ix_thrive_agent_run_event_created_at", "created_at"),
    )

    id = Column(Integer, primary_key=True)
    run_id = Column(String(36), nullable=False)
    seq = Column(Integer, nullable=False)
    event_type = Column(String(50), nullable=False)
    turn_index = Column(Integer, nullable=True)
    tool_call_id = Column(String(100), nullable=True)
    tool_name = Column(String(64), nullable=True)
    payload_json = Column(Text, nullable=True)
    payload_summary = Column(Text, nullable=True)
    payload_truncated = Column(Boolean, nullable=False, default=False)
    payload_bytes = Column(Integer, nullable=True)
    payload_hash = Column(String(64), nullable=True)
    elapsed_ms = Column(Integer, nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now(), nullable=False)


class AgentPatientAccess(Base):
    """Queryable record of every patient 'touch' for the Patient Access tab."""

    __tablename__ = "thrive_agent_patient_access"
    __table_args__ = (
        Index("ix_thrive_agent_patient_access_source", "source_id"),
        Index("ix_thrive_agent_patient_access_user", "user_id"),
        Index("ix_thrive_agent_patient_access_run", "run_id"),
        Index("ix_thrive_agent_patient_access_tool_call", "tool_call_id"),
        Index("ix_thrive_agent_patient_access_created", "created_at"),
        Index("ix_thrive_agent_patient_access_type", "access_type"),
    )

    id = Column(Integer, primary_key=True)
    run_id = Column(String(36), nullable=True)
    tool_call_id = Column(String(100), nullable=True)
    event_seq = Column(Integer, nullable=True)
    session_id = Column(String(64), nullable=False)
    user_id = Column(Integer, ForeignKey("thrive_user.id"), nullable=False)
    source_id = Column(String(50), nullable=False)
    display_name = Column(String(255), nullable=True)
    access_type = Column(String(40), nullable=False)
    access_origin = Column(String(40), nullable=False)
    tool_name = Column(String(64), nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now(), nullable=False)


class PatientSelectionEvent(Base):
    """One row each time a patient pin is set or cleared."""

    __tablename__ = "thrive_patient_selection_event"
    __table_args__ = (
        Index("ix_thrive_patient_selection_session", "session_id"),
        Index("ix_thrive_patient_selection_user", "user_id"),
        Index("ix_thrive_patient_selection_source", "source_id"),
        Index("ix_thrive_patient_selection_created", "created_at"),
    )

    id = Column(Integer, primary_key=True)
    session_id = Column(String(64), nullable=False)
    run_id = Column(String(36), nullable=True)
    user_id = Column(Integer, ForeignKey("thrive_user.id"), nullable=False)
    source_id = Column(String(50), nullable=True)
    previous_source_id = Column(String(50), nullable=True)
    display_name = Column(String(255), nullable=True)
    selection_origin = Column(String(32), nullable=False)
    action = Column(String(20), nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now(), nullable=False)


def seed_initial_data(session):
    # Seed User Roles
    roles_to_seed = [
        {"role_name": "Admin", "description": "Administrator with full access", "role": RoleTypeEnum.ADMIN},
        {
            "role_name": "Doctor",
            "description": "A physician who has the rights to view some individual patient data",
            "role": RoleTypeEnum.DOCTOR,
        },
        {
            "role_name": "Nurse",
            "description": "A nurse with access to patient data relevant to their duties",
            "role": RoleTypeEnum.NURSE,
        },
        {
            "role_name": "Patient",
            "description": "Patient access, only has access to see their own data or population data",
            "role": RoleTypeEnum.PATIENT,
        },
    ]

    for role_data in roles_to_seed:
        role = session.query(UserRole).filter_by(role_name=role_data["role_name"]).first()
        if not role:
            new_role = UserRole(**role_data)
            session.add(new_role)

    session.commit()  # Commit roles before users to ensure role IDs are available

    # Seed Users. Per Epic #179 email + organization are required (NOT NULL);
    # the seed values use the canonical thriveai.com domain and "ThriveAI"
    # organization so a fresh DB satisfies the constraint without admin
    # intervention. Existing dev DBs are migrated separately by the
    # require_user_email_org_role revision.
    users_to_seed = [
        {
            "username": "thriveai-kr",
            "first_name": "Kyle",
            "last_name": "Root",
            "email": "thriveai-kr@thriveai.com",
            "organization": "ThriveAI",
            "show_summary": True,
            "password": "5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8",
            "role_name": "Admin",
        },
        {
            "username": "thriveai-je",
            "first_name": "Joseph",
            "last_name": "Eberle",
            "email": "thriveai-je@thriveai.com",
            "organization": "ThriveAI",
            "show_summary": True,
            "password": "5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8",
            "role_name": "Admin",
        },
        {
            "username": "thriveai-as",
            "first_name": "Al",
            "last_name": "Seoud",
            "email": "thriveai-as@thriveai.com",
            "organization": "ThriveAI",
            "show_summary": True,
            "password": "5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8",
            "role_name": "Admin",
        },
        {
            "username": "thriveai-fm",
            "first_name": "Frank",
            "last_name": "Metty",
            "email": "thriveai-fm@thriveai.com",
            "organization": "ThriveAI",
            "show_summary": True,
            "password": "5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8",
            "role_name": "Admin",
        },
        {
            "username": "thriveai-dr",
            "first_name": "Dr.",
            "last_name": "Smith",
            "email": "thriveai-dr@thriveai.com",
            "organization": "ThriveAI",
            "show_summary": True,
            "password": "5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8",
            "role_name": "Doctor",
        },
        {
            "username": "thriveai-re",
            "first_name": "Rob",
            "last_name": "Enderle",
            "email": "thriveai-re@thriveai.com",
            "organization": "ThriveAI",
            "show_summary": True,
            "password": "5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8",
            "role_name": "Admin",
        },
    ]

    for user_data in users_to_seed:
        user = session.query(User).filter_by(username=user_data["username"]).first()
        if not user:
            role_name = user_data.pop("role_name")  # Remove role_name from user_data
            role = session.query(UserRole).filter_by(role_name=role_name).first()
            if role:  # Ensure role exists
                new_user = User(**user_data, user_role_id=role.id)
                session.add(new_user)
            else:
                logger.warning(f"Role '{role_name}' not found for user '{user_data['username']}'. User not created.")

    session.commit()


def init_db() -> None:
    """Run pending Alembic migrations and seed default data. Idempotent.

    Called once at app startup from `app.py`. Safe to call multiple times —
    Alembic's upgrade is a no-op when the DB is already at head, and
    `seed_initial_data` skips rows that already exist.
    """
    from alembic import command
    from alembic.config import Config

    if DATABASE_URL.startswith("sqlite:///"):
        db_path = Path(DATABASE_URL.removeprefix("sqlite:///"))
        db_path.parent.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parent.parent
    cfg = Config(str(repo_root / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", DATABASE_URL)
    command.upgrade(cfg, "head")

    session = SessionLocal()
    try:
        seed_initial_data(session)
    finally:
        session.close()
