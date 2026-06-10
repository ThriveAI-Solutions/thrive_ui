"""Helper functions for structured database logging.

This module provides functions to log:
- LLM context (RAG retrieval, prompts, token usage)
- User activity (login, logout, settings changes)
- Admin actions (user management, training approvals)
- Errors (structured error logging with context)
"""

import json
import traceback
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import func

from orm.models import (
    ActivityType,
    AdminAction,
    AdminActionType,
    ErrorCategory,
    ErrorLog,
    ErrorSeverity,
    LLMContext,
    SessionLocal,
    UserActivity,
)
from utils.error_fallback_sink import write_fallback_record
from utils.quick_logger import get_logger

logger = get_logger(__name__)


# ============== LLM Context Logging ==============


def log_llm_context(
    user_id: int | None,
    question: str,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    ddl_list: list[str] | None = None,
    doc_list: list[str] | None = None,
    question_sql_list: list[dict] | None = None,
    generated_sql: str | None = None,
    thinking_content: str | None = None,
    full_prompt: list | None = None,
    retrieval_time_ms: int | None = None,
    inference_time_ms: int | None = None,
    total_time_ms: int | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    message_id: int | None = None,
    group_id: str | None = None,
) -> LLMContext | None:
    """Log LLM context for SQL generation.

    Args:
        user_id: The ID of the user making the request
        question: The user's question
        llm_provider: Provider name (anthropic, ollama, gemini, etc.)
        llm_model: Model name (claude-3-5-sonnet, llama3.2, etc.)
        ddl_list: List of DDL statements retrieved from vector store
        doc_list: List of documentation snippets retrieved
        question_sql_list: List of {question, sql} example pairs
        generated_sql: The SQL that was generated
        thinking_content: Reasoning trace for thinking models
        full_prompt: Complete prompt sent to LLM (JSON-serializable)
        retrieval_time_ms: Time to retrieve RAG context
        inference_time_ms: Time for LLM to respond
        total_time_ms: Total end-to-end time
        input_tokens: Input token count (if available)
        output_tokens: Output token count (if available)
        message_id: FK to thrive_message if available
        group_id: Correlation UUID

    Returns:
        The created LLMContext record, or None if logging failed
    """
    ddl_list = ddl_list or []
    doc_list = doc_list or []
    question_sql_list = question_sql_list or []

    try:
        with SessionLocal() as session:
            context = LLMContext(
                message_id=message_id,
                group_id=group_id,
                user_id=user_id,
                question=question[:1000] if question else None,
                llm_provider=llm_provider,
                llm_model=llm_model,
                ddl_statements=json.dumps(ddl_list),
                documentation_snippets=json.dumps(doc_list),
                question_sql_examples=json.dumps(question_sql_list),
                ddl_count=len(ddl_list),
                doc_count=len(doc_list),
                example_count=len(question_sql_list),
                full_prompt=json.dumps(full_prompt) if full_prompt else None,
                generated_sql=generated_sql,
                thinking_content=thinking_content,
                retrieval_time_ms=retrieval_time_ms,
                inference_time_ms=inference_time_ms,
                total_time_ms=total_time_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=(input_tokens or 0) + (output_tokens or 0) if input_tokens or output_tokens else None,
            )
            session.add(context)
            session.commit()
            session.refresh(context)
            return context
    except Exception as e:
        logger.warning("Failed to log LLM context: %s", e)
        return None


def get_llm_context_for_message(message_id: int) -> LLMContext | None:
    """Get LLM context for a specific message."""
    try:
        with SessionLocal() as session:
            return session.query(LLMContext).filter(LLMContext.message_id == message_id).first()
    except Exception as e:
        logger.warning("Failed to get LLM context: %s", e)
        return None


def get_recent_llm_contexts(
    user_id: int | None = None,
    llm_provider: str | None = None,
    limit: int = 50,
) -> list[LLMContext]:
    """Get recent LLM context logs."""
    try:
        with SessionLocal() as session:
            query = session.query(LLMContext)
            if user_id:
                query = query.filter(LLMContext.user_id == user_id)
            if llm_provider:
                query = query.filter(LLMContext.llm_provider == llm_provider)
            return query.order_by(LLMContext.created_at.desc()).limit(limit).all()
    except Exception as e:
        logger.warning("Failed to get LLM contexts: %s", e)
        return []


# ============== User Activity Logging ==============


def log_user_activity(
    activity_type: ActivityType,
    description: str,
    user_id: int | None = None,
    username: str | None = None,
    old_value: dict | None = None,
    new_value: dict | None = None,
    ip_address: str | None = None,
    user_agent: str | None = None,
) -> UserActivity | None:
    """Log a user activity event.

    Args:
        activity_type: Type of activity from ActivityType enum
        description: Human-readable description of the activity
        user_id: The ID of the user (nullable for failed logins)
        username: Username (stored even if user lookup fails)
        old_value: Previous state (for settings changes)
        new_value: New state (for settings changes)
        ip_address: Client IP address
        user_agent: Client user agent string

    Returns:
        The created UserActivity record, or None if logging failed
    """
    try:
        with SessionLocal() as session:
            activity = UserActivity(
                user_id=user_id,
                username=username,
                activity_type=activity_type.value,
                description=description[:500] if description else None,
                old_value=json.dumps(old_value) if old_value else None,
                new_value=json.dumps(new_value) if new_value else None,
                ip_address=ip_address,
                user_agent=user_agent[:500] if user_agent else None,
            )
            session.add(activity)
            session.commit()
            session.refresh(activity)
            return activity
    except Exception as e:
        logger.warning("Failed to log user activity: %s", e)
        return None


def log_login(user_id: int | None, username: str, success: bool = True) -> UserActivity | None:
    """Log a login attempt.

    Args:
        user_id: User ID if login was successful
        username: Username attempted
        success: Whether login was successful

    Returns:
        The created UserActivity record
    """
    activity_type = ActivityType.LOGIN if success else ActivityType.LOGIN_FAILED
    description = f"User '{username}' logged in successfully" if success else f"Failed login attempt for '{username}'"
    return log_user_activity(
        activity_type=activity_type,
        description=description,
        user_id=user_id if success else None,
        username=username,
    )


def log_logout(user_id: int, username: str) -> UserActivity | None:
    """Log a logout event."""
    return log_user_activity(
        activity_type=ActivityType.LOGOUT,
        description=f"User '{username}' logged out",
        user_id=user_id,
        username=username,
    )


def log_settings_change(
    user_id: int,
    username: str,
    setting_name: str,
    old_val: Any,
    new_val: Any,
) -> UserActivity | None:
    """Log a user settings change."""
    return log_user_activity(
        activity_type=ActivityType.SETTINGS_CHANGE,
        description=f"Changed {setting_name}",
        user_id=user_id,
        username=username,
        old_value={setting_name: old_val},
        new_value={setting_name: new_val},
    )


def log_password_change(user_id: int, username: str) -> UserActivity | None:
    """Log a password change event."""
    return log_user_activity(
        activity_type=ActivityType.PASSWORD_CHANGE,
        description=f"User '{username}' changed their password",
        user_id=user_id,
        username=username,
    )


def get_user_activity_log(
    user_id: int | None = None,
    activity_type: ActivityType | None = None,
    days: int = 30,
    limit: int = 100,
) -> list[UserActivity]:
    """Get user activity log.

    Args:
        user_id: Filter by user ID
        activity_type: Filter by activity type
        days: Number of days to look back
        limit: Maximum number of records to return

    Returns:
        List of UserActivity records
    """
    try:
        since = datetime.now() - timedelta(days=days)
        with SessionLocal() as session:
            query = session.query(UserActivity).filter(UserActivity.created_at >= since)
            if user_id:
                query = query.filter(UserActivity.user_id == user_id)
            if activity_type:
                query = query.filter(UserActivity.activity_type == activity_type.value)
            return query.order_by(UserActivity.created_at.desc()).limit(limit).all()
    except Exception as e:
        logger.warning("Failed to get user activity log: %s", e)
        return []


# ============== Admin Action Logging ==============


def log_admin_action(
    admin_id: int,
    action_type: AdminActionType,
    description: str,
    target_user_id: int | None = None,
    target_username: str | None = None,
    target_entity_type: str | None = None,
    target_entity_id: str | None = None,
    old_value: dict | None = None,
    new_value: dict | None = None,
    affected_count: int | None = None,
    success: bool = True,
    error_message: str | None = None,
) -> AdminAction | None:
    """Log an admin action.

    Args:
        admin_id: ID of the admin performing the action
        action_type: Type of action from AdminActionType enum
        description: Human-readable description
        target_user_id: ID of user being acted upon (if applicable)
        target_username: Username (denormalized for deleted users)
        target_entity_type: Type of entity ('user', 'training_data', etc.)
        target_entity_id: ID of the entity
        old_value: Previous state
        new_value: New state
        affected_count: Number of records affected (for bulk ops)
        success: Whether the action succeeded
        error_message: Error message if action failed

    Returns:
        The created AdminAction record, or None if logging failed
    """
    try:
        with SessionLocal() as session:
            action = AdminAction(
                admin_id=admin_id,
                action_type=action_type.value,
                description=description[:500] if description else None,
                target_user_id=target_user_id,
                target_username=target_username,
                target_entity_type=target_entity_type,
                target_entity_id=str(target_entity_id) if target_entity_id else None,
                old_value=json.dumps(old_value) if old_value else None,
                new_value=json.dumps(new_value) if new_value else None,
                affected_count=affected_count,
                success=success,
                error_message=error_message[:500] if error_message else None,
            )
            session.add(action)
            session.commit()
            session.refresh(action)
            return action
    except Exception as e:
        logger.warning("Failed to log admin action: %s", e)
        return None


def get_admin_actions(
    admin_id: int | None = None,
    action_type: AdminActionType | None = None,
    target_user_id: int | None = None,
    days: int = 30,
    limit: int = 100,
) -> list[AdminAction]:
    """Get admin action log.

    Args:
        admin_id: Filter by admin who performed action
        action_type: Filter by action type
        target_user_id: Filter by target user
        days: Number of days to look back
        limit: Maximum number of records to return

    Returns:
        List of AdminAction records
    """
    try:
        since = datetime.now() - timedelta(days=days)
        with SessionLocal() as session:
            query = session.query(AdminAction).filter(AdminAction.created_at >= since)
            if admin_id:
                query = query.filter(AdminAction.admin_id == admin_id)
            if action_type:
                query = query.filter(AdminAction.action_type == action_type.value)
            if target_user_id:
                query = query.filter(AdminAction.target_user_id == target_user_id)
            return query.order_by(AdminAction.created_at.desc()).limit(limit).all()
    except Exception as e:
        logger.warning("Failed to get admin actions: %s", e)
        return []


# ============== Question Audit Trail ==============

MAX_AUDIT_EXPORT_ROWS = 50_000
_NO_ORG_SENTINEL = "(no org)"


def _question_audit_base_query(session, filters: dict):
    """Build the filtered base query over user-role Message rows joined to User.

    Returns a SQLAlchemy query that selects the rows in reverse-chronological
    order WITHOUT pagination. Used by both the paginated page helper and the
    export helper.
    """
    from sqlalchemy import or_

    from orm.models import Message, User
    from utils.enums import MessageType, RoleType

    days = int(filters.get("days") or 30)
    usernames = filters.get("usernames") or []
    orgs = filters.get("orgs") or []
    search = filters.get("search") or None

    since = datetime.utcnow() - timedelta(days=days)

    query = (
        session.query(
            Message.id.label("user_message_id"),
            Message.content.label("question"),
            Message.created_at.label("asked_at"),
            User.id.label("user_id"),
            User.username.label("username"),
            User.organization.label("organization"),
        )
        .join(User, User.id == Message.user_id)
        .filter(Message.role == RoleType.USER.value, Message.created_at >= since)
    )

    if usernames:
        query = query.filter(User.username.in_(usernames))

    if orgs:
        org_clauses = []
        concrete = [o for o in orgs if o != _NO_ORG_SENTINEL]
        if _NO_ORG_SENTINEL in orgs:
            org_clauses.append(User.organization.is_(None))
            org_clauses.append(User.organization == "")
        if concrete:
            org_clauses.append(User.organization.in_(concrete))
        if org_clauses:
            query = query.filter(or_(*org_clauses))

    if search:
        from sqlalchemy import exists
        from sqlalchemy.orm import aliased

        s = f"%{search.lower()}%"
        SqlMsg = aliased(Message)
        sql_exists = exists().where(
            (SqlMsg.role == RoleType.ASSISTANT.value)
            & (SqlMsg.type == MessageType.SQL.value)
            & (SqlMsg.question == Message.content)
            & (SqlMsg.user_id == Message.user_id)
            & (SqlMsg.content.ilike(s))
        )
        query = query.filter(or_(Message.content.ilike(s), sql_exists))

    return query.order_by(Message.created_at.desc())


def _enrich_with_assistant_aggregates(session, base_rows: list) -> list[dict]:
    """For each base row, attach sql_text / summary_text / dataframe_preview /
    error_text (most recent assistant message of each type) and elapsed_seconds
    (sum of assistant elapsed_time across all assistant rows for the question)."""
    from orm.models import Message
    from utils.enums import MessageType, RoleType

    if not base_rows:
        return []

    user_ids = {r.user_id for r in base_rows}
    questions = {r.question for r in base_rows}

    assistant_rows = (
        session.query(
            Message.user_id.label("user_id"),
            Message.question.label("question"),
            Message.type.label("type"),
            Message.content.label("content"),
            Message.created_at.label("created_at"),
            Message.elapsed_time.label("elapsed_time"),
        )
        .filter(
            Message.role == RoleType.ASSISTANT.value,
            Message.user_id.in_(user_ids),
            Message.question.in_(questions),
        )
        .order_by(Message.created_at.desc())
        .all()
    )

    chart_types = {
        MessageType.PLOTLY_CHART.value,
        MessageType.ST_LINE_CHART.value,
        MessageType.ST_BAR_CHART.value,
        MessageType.ST_AREA_CHART.value,
        MessageType.ST_SCATTER_CHART.value,
    }

    by_key: dict[tuple[int, str], dict] = {}
    for ar in assistant_rows:
        key = (ar.user_id, ar.question)
        acc = by_key.setdefault(
            key,
            {
                "sql_text": None,
                "summary_text": None,
                "dataframe_preview": None,
                "error_text": None,
                "elapsed_seconds": 0.0,
                "has_chart": False,
            },
        )
        if ar.elapsed_time is not None:
            try:
                acc["elapsed_seconds"] += float(ar.elapsed_time)
            except (TypeError, ValueError):
                pass
        # Rows arrive newest-first; keep the first (newest) per type.
        if ar.type == MessageType.SQL.value and acc["sql_text"] is None:
            acc["sql_text"] = ar.content
        elif ar.type == MessageType.SUMMARY.value and acc["summary_text"] is None:
            acc["summary_text"] = ar.content
        elif ar.type == MessageType.DATAFRAME.value and acc["dataframe_preview"] is None:
            acc["dataframe_preview"] = ar.content
        elif ar.type == MessageType.ERROR.value and acc["error_text"] is None:
            acc["error_text"] = ar.content
        elif ar.type in chart_types:
            acc["has_chart"] = True

    items = []
    for r in base_rows:
        key = (r.user_id, r.question)
        agg = by_key.get(key) or {
            "sql_text": None,
            "summary_text": None,
            "dataframe_preview": None,
            "error_text": None,
            "elapsed_seconds": 0.0,
            "has_chart": False,
        }
        has_error = bool(agg["error_text"])
        has_success_artifact = bool(agg["summary_text"]) or bool(agg["dataframe_preview"]) or agg["has_chart"]
        if has_error:
            status = "Error"
        elif has_success_artifact:
            status = "Success"
        else:
            status = "Empty"
        items.append(
            {
                "asked_at": r.asked_at,
                "user_id": r.user_id,
                "username": r.username,
                "organization": r.organization,
                "question": r.question,
                "sql_text": agg["sql_text"],
                "status": status,
                "elapsed_seconds": float(agg["elapsed_seconds"]),
                "summary_text": agg["summary_text"],
                "dataframe_preview": agg["dataframe_preview"],
                "error_text": agg["error_text"],
                "user_message_id": r.user_message_id,
            }
        )
    return items


def get_question_audit_page(filters: dict, page: int = 1, page_size: int = 50) -> dict:
    """Paginated question audit. Returns {"items": [...], "total": int}.

    See Feature #134 spec for filter semantics and item shape.
    """
    try:
        with SessionLocal() as session:
            base_query = _question_audit_base_query(session, filters)
            total = base_query.with_entities(func.count()).order_by(None).scalar() or 0
            offset = max(0, (int(page) - 1) * int(page_size))
            rows = base_query.offset(offset).limit(int(page_size)).all()
            items = _enrich_with_assistant_aggregates(session, rows)
            return {"items": items, "total": int(total)}
    except Exception as e:
        logger.warning("get_question_audit_page failed: %s", e)
        return {"items": [], "total": 0}


def get_question_audit_filter_options(days: int = 30) -> dict:
    """Return distinct {usernames, orgs} for users with at least one
    user-role Message in the date range. Lists sorted ascending. Includes
    ``(no org)`` sentinel if any in-range user has NULL/empty organization."""
    try:
        from orm.models import Message, User
        from utils.enums import RoleType

        since = datetime.utcnow() - timedelta(days=int(days))
        with SessionLocal() as session:
            rows = (
                session.query(User.username, User.organization)
                .join(Message, Message.user_id == User.id)
                .filter(Message.role == RoleType.USER.value, Message.created_at >= since)
                .distinct()
                .all()
            )
        usernames = sorted({r[0] for r in rows if r[0]})
        orgs_raw = {r[1] for r in rows}
        concrete_orgs = sorted({o for o in orgs_raw if o})
        has_no_org = any((o is None) or (o == "") for o in orgs_raw)
        orgs = ([_NO_ORG_SENTINEL] if has_no_org else []) + concrete_orgs
        return {"usernames": usernames, "orgs": orgs}
    except Exception as e:
        logger.warning("get_question_audit_filter_options failed: %s", e)
        return {"usernames": [], "orgs": []}


def get_question_audit_export(filters: dict) -> list[dict]:
    """Full filtered set with no pagination, capped at MAX_AUDIT_EXPORT_ROWS."""
    try:
        with SessionLocal() as session:
            base_query = _question_audit_base_query(session, filters)
            rows = base_query.limit(MAX_AUDIT_EXPORT_ROWS + 1).all()
            if len(rows) > MAX_AUDIT_EXPORT_ROWS:
                logger.warning(
                    "get_question_audit_export hit MAX_AUDIT_EXPORT_ROWS=%s; truncating",
                    MAX_AUDIT_EXPORT_ROWS,
                )
                rows = rows[:MAX_AUDIT_EXPORT_ROWS]
            return _enrich_with_assistant_aggregates(session, rows)
    except Exception as e:
        logger.warning("get_question_audit_export failed: %s", e)
        return []


# ============== Error Logging ==============


def log_error(
    category: ErrorCategory,
    severity: ErrorSeverity,
    error_type: str,
    error_message: str,
    user_id: int | None = None,
    message_id: int | None = None,
    group_id: str | None = None,
    question: str | None = None,
    generated_sql: str | None = None,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    context_data: dict | None = None,
    include_traceback: bool = True,
    auto_retry_attempted: bool = False,
    retry_successful: bool | None = None,
    retry_count: int = 0,
) -> ErrorLog | None:
    """Log an error to the database.

    Args:
        category: Error category from ErrorCategory enum
        severity: Error severity from ErrorSeverity enum
        error_type: Exception class name
        error_message: Error message text
        user_id: ID of user who triggered the error
        message_id: FK to thrive_message if available
        group_id: Correlation UUID
        question: User question that triggered the error
        generated_sql: SQL that failed (if applicable)
        llm_provider: LLM provider being used
        llm_model: LLM model being used
        context_data: Additional debugging context (JSON-serializable)
        include_traceback: Whether to include full stack trace
        auto_retry_attempted: Whether an automatic retry was attempted
        retry_successful: Whether the retry succeeded
        retry_count: Number of retry attempts

    Returns:
        The created ErrorLog record, or None if logging failed
    """
    captured_traceback = traceback.format_exc() if include_traceback else None
    try:
        with SessionLocal() as session:
            error = ErrorLog(
                user_id=user_id,
                message_id=message_id,
                group_id=group_id,
                category=category.value,
                severity=severity.value,
                error_type=error_type,
                error_message=error_message[:2000] if error_message else None,
                stack_trace=captured_traceback,
                question=question[:1000] if question else None,
                generated_sql=generated_sql,
                llm_provider=llm_provider,
                llm_model=llm_model,
                context_data=json.dumps(context_data) if context_data else None,
                auto_retry_attempted=auto_retry_attempted,
                retry_successful=retry_successful,
                retry_count=retry_count,
            )
            session.add(error)
            session.commit()
            session.refresh(error)
            return error
    except Exception as e:
        try:
            write_fallback_record(
                {
                    "created_at": datetime.now().isoformat(),
                    "category": category.value,
                    "severity": severity.value,
                    "error_type": error_type,
                    "error_message": error_message[:2000] if error_message else None,
                    "stack_trace": captured_traceback,
                    "question": question[:1000] if question else None,
                    "generated_sql": generated_sql,
                    "llm_provider": llm_provider,
                    "llm_model": llm_model,
                    "context_data": json.dumps(context_data) if context_data else None,
                    "user_id": user_id,
                    "message_id": message_id,
                    "group_id": group_id,
                    "auto_retry_attempted": auto_retry_attempted,
                    "retry_successful": retry_successful,
                    "retry_count": retry_count,
                }
            )
        except Exception:
            pass  # write_fallback_record is already defensive; belt-and-braces
        logger.warning("Failed to log error: %s", e)
        return None


def log_sql_generation_error(
    error: Exception,
    question: str,
    user_id: int | None = None,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    group_id: str | None = None,
) -> ErrorLog | None:
    """Convenience method for SQL generation errors."""
    return log_error(
        category=ErrorCategory.SQL_GENERATION,
        severity=ErrorSeverity.ERROR,
        error_type=type(error).__name__,
        error_message=str(error),
        user_id=user_id,
        question=question,
        llm_provider=llm_provider,
        llm_model=llm_model,
        group_id=group_id,
    )


def log_sql_execution_error(
    error: Exception,
    sql: str,
    question: str | None = None,
    user_id: int | None = None,
    message_id: int | None = None,
    group_id: str | None = None,
    retry_count: int = 0,
    auto_retry_attempted: bool = False,
    retry_successful: bool | None = None,
) -> ErrorLog | None:
    """Convenience method for SQL execution errors."""
    return log_error(
        category=ErrorCategory.SQL_EXECUTION,
        severity=ErrorSeverity.ERROR,
        error_type=type(error).__name__,
        error_message=str(error),
        user_id=user_id,
        message_id=message_id,
        group_id=group_id,
        question=question,
        generated_sql=sql,
        auto_retry_attempted=auto_retry_attempted,
        retry_successful=retry_successful,
        retry_count=retry_count,
    )


def log_chart_generation_error(
    error: Exception,
    user_id: int | None = None,
    question: str | None = None,
    group_id: str | None = None,
) -> ErrorLog | None:
    """Convenience method for chart generation errors."""
    return log_error(
        category=ErrorCategory.CHART_GENERATION,
        severity=ErrorSeverity.ERROR,
        error_type=type(error).__name__,
        error_message=str(error),
        user_id=user_id,
        question=question,
        group_id=group_id,
    )


def get_recent_errors(
    user_id: int | None = None,
    category: ErrorCategory | None = None,
    severity: ErrorSeverity | None = None,
    days: int = 7,
    limit: int = 50,
) -> list[ErrorLog]:
    """Get recent errors, optionally filtered.

    Args:
        user_id: Filter by user ID
        category: Filter by error category
        severity: Filter by severity level
        days: Number of days to look back
        limit: Maximum number of records to return

    Returns:
        List of ErrorLog records
    """
    try:
        since = datetime.now() - timedelta(days=days)
        with SessionLocal() as session:
            query = session.query(ErrorLog).filter(ErrorLog.created_at >= since)
            if user_id:
                query = query.filter(ErrorLog.user_id == user_id)
            if category:
                query = query.filter(ErrorLog.category == category.value)
            if severity:
                query = query.filter(ErrorLog.severity == severity.value)
            return query.order_by(ErrorLog.created_at.desc()).limit(limit).all()
    except Exception as e:
        logger.warning("Failed to get recent errors: %s", e)
        return []


# ============== Analytics Query Functions ==============


def get_llm_stats(days: int = 30) -> dict:
    """Get aggregate LLM context statistics for analytics."""
    try:
        since = datetime.now() - timedelta(days=days)
        with SessionLocal() as session:
            from sqlalchemy import func as sqla_func

            base = session.query(LLMContext).filter(LLMContext.created_at >= since)
            total = base.count() or 0
            avg_latency = (
                session.query(sqla_func.avg(LLMContext.total_time_ms)).filter(LLMContext.created_at >= since).scalar()
                or 0
            )
            avg_tokens = (
                session.query(sqla_func.avg(LLMContext.total_tokens))
                .filter(LLMContext.created_at >= since, LLMContext.total_tokens.isnot(None))
                .scalar()
                or 0
            )
            avg_ddl = (
                session.query(sqla_func.avg(LLMContext.ddl_count)).filter(LLMContext.created_at >= since).scalar() or 0
            )
            avg_doc = (
                session.query(sqla_func.avg(LLMContext.doc_count)).filter(LLMContext.created_at >= since).scalar() or 0
            )
            avg_example = (
                session.query(sqla_func.avg(LLMContext.example_count)).filter(LLMContext.created_at >= since).scalar()
                or 0
            )
            return {
                "total": total,
                "avg_latency_ms": float(avg_latency),
                "avg_tokens": float(avg_tokens),
                "avg_ddl_count": float(avg_ddl),
                "avg_doc_count": float(avg_doc),
                "avg_example_count": float(avg_example),
            }
    except Exception as e:
        logger.warning("Failed to get LLM stats: %s", e)
        return {
            "total": 0,
            "avg_latency_ms": 0,
            "avg_tokens": 0,
            "avg_ddl_count": 0,
            "avg_doc_count": 0,
            "avg_example_count": 0,
        }


def get_llm_over_time(days: int = 30) -> list[dict]:
    """Get LLM context counts over time for charting."""
    try:
        since = datetime.now() - timedelta(days=days)
        with SessionLocal() as session:
            from sqlalchemy import func as sqla_func

            date_expr = sqla_func.strftime("%Y-%m-%d", LLMContext.created_at)
            results = (
                session.query(
                    date_expr.label("date"),
                    sqla_func.count(LLMContext.id).label("queries"),
                    sqla_func.avg(LLMContext.total_time_ms).label("avg_latency"),
                )
                .filter(LLMContext.created_at >= since)
                .group_by(date_expr)
                .order_by(date_expr)
                .all()
            )
            return [{"date": r.date, "queries": r.queries, "avg_latency": float(r.avg_latency or 0)} for r in results]
    except Exception as e:
        logger.warning("Failed to get LLM over time: %s", e)
        return []


def get_llm_provider_breakdown(days: int = 30) -> list[dict]:
    """Get query counts by provider and model."""
    try:
        since = datetime.now() - timedelta(days=days)
        with SessionLocal() as session:
            from sqlalchemy import func as sqla_func

            results = (
                session.query(
                    LLMContext.llm_provider,
                    LLMContext.llm_model,
                    sqla_func.count(LLMContext.id).label("count"),
                    sqla_func.avg(LLMContext.total_time_ms).label("avg_latency"),
                )
                .filter(LLMContext.created_at >= since)
                .group_by(LLMContext.llm_provider, LLMContext.llm_model)
                .order_by(sqla_func.count(LLMContext.id).desc())
                .all()
            )
            return [
                {
                    "provider": r.llm_provider or "unknown",
                    "model": r.llm_model or "unknown",
                    "count": r.count,
                    "avg_latency": float(r.avg_latency or 0),
                }
                for r in results
            ]
    except Exception as e:
        logger.warning("Failed to get LLM provider breakdown: %s", e)
        return []


def get_recent_llm_queries(days: int = 7, limit: int = 50) -> list[dict]:
    """Get recent LLM queries with context details for drill-down."""
    try:
        since = datetime.now() - timedelta(days=days)
        with SessionLocal() as session:
            results = (
                session.query(LLMContext)
                .filter(LLMContext.created_at >= since)
                .order_by(LLMContext.created_at.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "id": r.id,
                    "created_at": r.created_at,
                    "question": r.question,
                    "provider": r.llm_provider,
                    "model": r.llm_model,
                    "total_time_ms": r.total_time_ms,
                    "total_tokens": r.total_tokens,
                    "ddl_count": r.ddl_count,
                    "doc_count": r.doc_count,
                    "example_count": r.example_count,
                    "generated_sql": r.generated_sql,
                    "ddl_statements": r.ddl_statements,
                    "documentation_snippets": r.documentation_snippets,
                    "question_sql_examples": r.question_sql_examples,
                }
                for r in results
            ]
    except Exception as e:
        logger.warning("Failed to get recent LLM queries: %s", e)
        return []


def get_activity_stats(days: int = 30) -> dict:
    """Get user activity statistics."""
    try:
        since = datetime.now() - timedelta(days=days)
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        with SessionLocal() as session:
            from sqlalchemy import func as sqla_func

            logins_today = (
                session.query(sqla_func.count(UserActivity.id))
                .filter(
                    UserActivity.activity_type == ActivityType.LOGIN.value,
                    UserActivity.created_at >= today_start,
                )
                .scalar()
                or 0
            )
            failed_logins = (
                session.query(sqla_func.count(UserActivity.id))
                .filter(
                    UserActivity.activity_type == ActivityType.LOGIN_FAILED.value,
                    UserActivity.created_at >= since,
                )
                .scalar()
                or 0
            )
            settings_changes = (
                session.query(sqla_func.count(UserActivity.id))
                .filter(
                    UserActivity.activity_type == ActivityType.SETTINGS_CHANGE.value,
                    UserActivity.created_at >= since,
                )
                .scalar()
                or 0
            )
            unique_users = (
                session.query(sqla_func.count(sqla_func.distinct(UserActivity.user_id)))
                .filter(
                    UserActivity.activity_type == ActivityType.LOGIN.value,
                    UserActivity.created_at >= since,
                )
                .scalar()
                or 0
            )
            return {
                "logins_today": logins_today,
                "failed_logins": failed_logins,
                "settings_changes": settings_changes,
                "unique_users": unique_users,
            }
    except Exception as e:
        logger.warning("Failed to get activity stats: %s", e)
        return {"logins_today": 0, "failed_logins": 0, "settings_changes": 0, "unique_users": 0}


def get_activity_over_time(days: int = 30) -> list[dict]:
    """Get login activity over time."""
    try:
        since = datetime.now() - timedelta(days=days)
        with SessionLocal() as session:
            from sqlalchemy import case
            from sqlalchemy import func as sqla_func

            date_expr = sqla_func.strftime("%Y-%m-%d", UserActivity.created_at)
            results = (
                session.query(
                    date_expr.label("date"),
                    sqla_func.sum(case((UserActivity.activity_type == ActivityType.LOGIN.value, 1), else_=0)).label(
                        "logins"
                    ),
                    sqla_func.sum(
                        case((UserActivity.activity_type == ActivityType.LOGIN_FAILED.value, 1), else_=0)
                    ).label("failed_logins"),
                )
                .filter(UserActivity.created_at >= since)
                .group_by(date_expr)
                .order_by(date_expr)
                .all()
            )
            return [
                {"date": r.date, "logins": int(r.logins or 0), "failed_logins": int(r.failed_logins or 0)}
                for r in results
            ]
    except Exception as e:
        logger.warning("Failed to get activity over time: %s", e)
        return []


def get_activity_by_type(days: int = 30) -> list[dict]:
    """Get activity counts by type."""
    try:
        since = datetime.now() - timedelta(days=days)
        with SessionLocal() as session:
            from sqlalchemy import func as sqla_func

            results = (
                session.query(
                    UserActivity.activity_type,
                    sqla_func.count(UserActivity.id).label("count"),
                )
                .filter(UserActivity.created_at >= since)
                .group_by(UserActivity.activity_type)
                .all()
            )
            return [{"activity_type": r.activity_type, "count": r.count} for r in results]
    except Exception as e:
        logger.warning("Failed to get activity by type: %s", e)
        return []


def get_user_activity_page(days: int = 7, page: int = 1, page_size: int = 50) -> dict:
    """Paginated user activity for the audit log. Returns {"items": [...], "total": int}.

    Each item exposes the full UserActivity row (10 fields): ``id``, ``created_at``,
    ``user_id``, ``username``, ``activity_type``, ``description``, ``old_value``,
    ``new_value``, ``ip_address``, ``user_agent``. See Feature #157 spec.

    ``user_id`` can be null for failed-login rows where user lookup failed.
    """
    try:
        since = datetime.now() - timedelta(days=days)
        with SessionLocal() as session:
            base_query = session.query(UserActivity).filter(UserActivity.created_at >= since)
            total = base_query.with_entities(func.count(UserActivity.id)).scalar() or 0
            offset = max(0, (int(page) - 1) * int(page_size))
            results = base_query.order_by(UserActivity.created_at.desc()).offset(offset).limit(int(page_size)).all()
            items = [
                {
                    "id": r.id,
                    "created_at": r.created_at,
                    "user_id": r.user_id,
                    "username": r.username,
                    "activity_type": r.activity_type,
                    "description": r.description,
                    "old_value": r.old_value,
                    "new_value": r.new_value,
                    "ip_address": r.ip_address,
                    "user_agent": r.user_agent,
                }
                for r in results
            ]
            return {"items": items, "total": int(total)}
    except Exception as e:
        logger.warning("get_user_activity_page failed: %s", e)
        return {"items": [], "total": 0}


def get_error_stats(days: int = 7) -> dict:
    """Get error statistics for analytics."""
    try:
        since = datetime.now() - timedelta(days=days)
        with SessionLocal() as session:
            from sqlalchemy import func as sqla_func

            total = session.query(sqla_func.count(ErrorLog.id)).filter(ErrorLog.created_at >= since).scalar() or 0
            critical = (
                session.query(sqla_func.count(ErrorLog.id))
                .filter(
                    ErrorLog.created_at >= since,
                    ErrorLog.severity == ErrorSeverity.CRITICAL.value,
                )
                .scalar()
                or 0
            )
            sql_errors = (
                session.query(sqla_func.count(ErrorLog.id))
                .filter(
                    ErrorLog.created_at >= since,
                    ErrorLog.category.in_([ErrorCategory.SQL_GENERATION.value, ErrorCategory.SQL_EXECUTION.value]),
                )
                .scalar()
                or 0
            )
            retry_attempted = (
                session.query(sqla_func.count(ErrorLog.id))
                .filter(ErrorLog.created_at >= since, ErrorLog.auto_retry_attempted == True)  # noqa: E712
                .scalar()
                or 0
            )
            retry_successful = (
                session.query(sqla_func.count(ErrorLog.id))
                .filter(ErrorLog.created_at >= since, ErrorLog.retry_successful == True)  # noqa: E712
                .scalar()
                or 0
            )
            retry_rate = (retry_successful / retry_attempted * 100) if retry_attempted > 0 else 0

            return {
                "total": total,
                "critical": critical,
                "sql_errors": sql_errors,
                "retry_success_rate": round(retry_rate, 1),
            }
    except Exception as e:
        logger.warning("Failed to get error stats: %s", e)
        return {"total": 0, "critical": 0, "sql_errors": 0, "retry_success_rate": 0}


def get_admin_action_stats(days: int = 30) -> dict:
    """Get admin action statistics."""
    try:
        since = datetime.now() - timedelta(days=days)
        with SessionLocal() as session:
            from sqlalchemy import func as sqla_func

            total = session.query(sqla_func.count(AdminAction.id)).filter(AdminAction.created_at >= since).scalar() or 0
            user_changes = (
                session.query(sqla_func.count(AdminAction.id))
                .filter(
                    AdminAction.created_at >= since,
                    AdminAction.action_type.in_(
                        [
                            AdminActionType.USER_CREATE.value,
                            AdminActionType.USER_UPDATE.value,
                            AdminActionType.USER_DELETE.value,
                            AdminActionType.USER_PASSWORD_RESET.value,
                        ]
                    ),
                )
                .scalar()
                or 0
            )
            training_actions = (
                session.query(sqla_func.count(AdminAction.id))
                .filter(
                    AdminAction.created_at >= since,
                    AdminAction.action_type.in_(
                        [
                            AdminActionType.TRAINING_APPROVE.value,
                            AdminActionType.TRAINING_REJECT.value,
                            AdminActionType.TRAINING_BULK_APPROVE.value,
                            AdminActionType.TRAINING_BULK_REJECT.value,
                        ]
                    ),
                )
                .scalar()
                or 0
            )
            failed = (
                session.query(sqla_func.count(AdminAction.id))
                .filter(AdminAction.created_at >= since, AdminAction.success == False)  # noqa: E712
                .scalar()
                or 0
            )
            return {
                "total": total,
                "user_changes": user_changes,
                "training_actions": training_actions,
                "failed": failed,
            }
    except Exception as e:
        logger.warning("Failed to get admin action stats: %s", e)
        return {"total": 0, "user_changes": 0, "training_actions": 0, "failed": 0}


def get_admin_actions_by_type(days: int = 30) -> list[dict]:
    """Get admin action counts by type."""
    try:
        since = datetime.now() - timedelta(days=days)
        with SessionLocal() as session:
            from sqlalchemy import func as sqla_func

            results = (
                session.query(
                    AdminAction.action_type,
                    sqla_func.count(AdminAction.id).label("count"),
                )
                .filter(AdminAction.created_at >= since)
                .group_by(AdminAction.action_type)
                .all()
            )
            return [{"action_type": r.action_type, "count": r.count} for r in results]
    except Exception as e:
        logger.warning("Failed to get admin actions by type: %s", e)
        return []


def get_recent_admin_actions(days: int = 30, limit: int = 50) -> list[dict]:
    """Get recent admin actions for audit log."""
    try:
        since = datetime.now() - timedelta(days=days)
        with SessionLocal() as session:
            from orm.models import User

            results = (
                session.query(AdminAction, User.username.label("admin_username"))
                .outerjoin(User, User.id == AdminAction.admin_id)
                .filter(AdminAction.created_at >= since)
                .order_by(AdminAction.created_at.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "id": r.AdminAction.id,
                    "created_at": r.AdminAction.created_at,
                    "admin_username": r.admin_username or "Unknown",
                    "action_type": r.AdminAction.action_type,
                    "target_username": r.AdminAction.target_username,
                    "description": r.AdminAction.description,
                    "success": r.AdminAction.success,
                    "old_value": r.AdminAction.old_value,
                    "new_value": r.AdminAction.new_value,
                }
                for r in results
            ]
    except Exception as e:
        logger.warning("Failed to get recent admin actions: %s", e)
        return []
