import enum
import json
import logging
from decimal import Decimal

import pandas as pd
import streamlit as st
from sqlalchemy import TIMESTAMP, Boolean, Column, ForeignKey, Index, Integer, Numeric, String, create_engine, func
from sqlalchemy import Enum as SqlEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

from utils.enums import MessageType, RoleType, ThemeType

logger = logging.getLogger(__name__)

# Load database settings from st.secrets
# db_settings = st.secrets["postgres"]
db_settings = st.secrets.get("sqlite", {"database": "./pgDatabase/db.sqlite3"})

# Construct the database URL
# DATABASE_URL = f"postgresql://{db_settings['user']}:{db_settings['password']}@{db_settings['host']}:{db_settings['port']}/{db_settings['database']}"
DATABASE_URL = f"sqlite:///{db_settings['database']}"

# Create the SQLAlchemy engine
engine = create_engine(DATABASE_URL)

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
    user_role_id = Column(Integer, ForeignKey("thrive_user_role.id"))
    username = Column(String(50), nullable=False, unique=True)
    first_name = Column(String(50), nullable=False)
    last_name = Column(String(50), nullable=False)
    password = Column(String(255), nullable=False)
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
    min_message_id = Column(Integer, default=0)
    theme = Column(String(50), default=ThemeType.HEALTHELINK.value)
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
            "min_message_id": self.min_message_id,
            "theme": self.theme,
            "selected_llm_provider": self.selected_llm_provider,
            "selected_llm_model": self.selected_llm_model,
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

    # Seed Users
    users_to_seed = [
        {
            "username": "thriveai-kr",
            "first_name": "Kyle",
            "last_name": "Root",
            "show_summary": True,
            "password": "5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8",
            "role_name": "Admin",
        },
        {
            "username": "thriveai-je",
            "first_name": "Joseph",
            "last_name": "Eberle",
            "show_summary": True,
            "password": "5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8",
            "role_name": "Admin",
        },
        {
            "username": "thriveai-as",
            "first_name": "Al",
            "last_name": "Seoud",
            "show_summary": True,
            "password": "5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8",
            "role_name": "Admin",
        },
        {
            "username": "thriveai-fm",
            "first_name": "Frank",
            "last_name": "Metty",
            "show_summary": True,
            "password": "5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8",
            "role_name": "Admin",
        },
        {
            "username": "thriveai-dr",
            "first_name": "Dr.",
            "last_name": "Smith",
            "show_summary": True,
            "password": "5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8",
            "role_name": "Doctor",
        },
        {
            "username": "thriveai-re",
            "first_name": "Rob",
            "last_name": "Enderle",
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


# Create tables
Base.metadata.create_all(bind=engine)

# Seed initial data
db_session = SessionLocal()
seed_initial_data(db_session)
db_session.close()
