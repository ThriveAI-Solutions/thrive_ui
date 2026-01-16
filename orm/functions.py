import hashlib
import json
import logging

import streamlit as st
from sqlalchemy import case, func
from sqlalchemy.orm import joinedload

from orm.models import Message, SessionLocal, User, UserRole
from utils.enums import MessageType, RoleType

logger = logging.getLogger(__name__)


def _get_current_user_id() -> int | None:
    """Get the current user ID from session state cookies."""
    try:
        user_id_str = st.session_state.cookies.get("user_id")
        if user_id_str:
            return json.loads(user_id_str)
    except Exception:
        pass
    return None


def verify_user_credentials(username: str, password: str) -> bool:
    try:
        # Create a new database session
        with SessionLocal() as session:
            # Hash the password using SHA-256
            hashed_password = hashlib.sha256(password.encode()).hexdigest()

            # Query to check if the username and hashed password exist in the users table
            user = (
                session.query(User)
                .filter(func.lower(User.username) == username.lower(), User.password == hashed_password)
                .one_or_none()
            )

            if user:
                st.session_state.cookies["user_id"] = json.dumps(user.id)
                userRole = session.query(UserRole).filter(UserRole.id == user.user_role_id).one_or_none()
                st.session_state.cookies["role_name"] = userRole.role_name
                st.session_state.user_role = userRole.role.value

                # Log successful login
                try:
                    from orm.logging_functions import log_login

                    log_login(user_id=user.id, username=username, success=True)
                except Exception as e:
                    logger.warning("Failed to log successful login for user %s: %s", username, e)
            else:
                # Log failed login attempt
                try:
                    from orm.logging_functions import log_login

                    log_login(user_id=None, username=username, success=False)
                except Exception as e:
                    logger.warning("Failed to log failed login attempt for user %s: %s", username, e)

            # Return True if the user exists, otherwise return False
            return user is not None
    except Exception as e:
        st.error(f"Error verifying user credentials: {e}")
        logger.error(f"Error verifying user credentials: {e}")
        return False


def change_password(user_id: int, current_password: str, new_password: str) -> bool:
    try:
        # Create a new database session
        with SessionLocal() as session:
            # Retrieve the existing user from the database
            user = session.query(User).filter(User.id == user_id).first()

            current_password_hashed = hashlib.sha256(current_password.encode()).hexdigest()

            # Verify the current password
            if user and current_password_hashed == user.password:
                # Update the user's password in the database
                user.password = hashlib.sha256(new_password.encode()).hexdigest()
                session.commit()

                # Log password change
                try:
                    from orm.logging_functions import log_password_change

                    log_password_change(user_id=user_id, username=user.username)
                except Exception as e:
                    logger.warning("Failed to log password change for user %s: %s", user_id, e)

                return True
            else:
                return False
    except Exception as e:
        st.error(f"Error changing password: {e}")
        logger.error(f"Error changing password: {e}")
        return False


def set_user_preferences_in_session_state():
    try:
        user_id_str = st.session_state.cookies.get("user_id")
        if not user_id_str:
            return None

        user_id = json.loads(user_id_str)
        user = get_user(user_id)

        # if "loaded" not in st.session_state:
        st.session_state.show_sql = user.show_sql
        st.session_state.show_table = user.show_table
        st.session_state.show_plotly_code = user.show_plotly_code
        st.session_state.show_chart = user.show_chart
        st.session_state.show_question_history = user.show_question_history
        st.session_state.show_summary = user.show_summary
        st.session_state.voice_input = user.voice_input
        st.session_state.speak_summary = user.speak_summary
        st.session_state.show_suggested = user.show_suggested
        st.session_state.show_followup = user.show_followup
        st.session_state.show_elapsed_time = user.show_elapsed_time
        st.session_state.llm_fallback = user.llm_fallback
        st.session_state.min_message_id = user.min_message_id
        st.session_state.user_role = user.role.role.value
        st.session_state.user_theme = user.theme
        st.session_state.selected_llm_provider = user.selected_llm_provider
        st.session_state.selected_llm_model = user.selected_llm_model
        st.session_state.loaded = True  # dont call after initial load
        st.session_state.username = f"{user.first_name} {user.last_name}"

        return user
    except Exception as e:
        st.error(f"Error setting user preferences in session state: {e}")
        logger.error(f"Error setting user preferences in session state: {e}")


def save_user_settings():
    try:
        user_id = st.session_state.cookies.get("user_id")
        user_id = json.loads(user_id)

        # Create a new database session
        session = SessionLocal()

        # Retrieve the existing user from the database
        user = session.query(User).filter(User.id == user_id).first()

        if user:
            setattr(user, "show_sql", st.session_state.show_sql)
            setattr(user, "show_table", st.session_state.show_table)
            setattr(user, "show_plotly_code", st.session_state.show_plotly_code)
            setattr(user, "show_chart", st.session_state.show_chart)
            setattr(user, "show_question_history", st.session_state.show_question_history)
            setattr(user, "show_summary", st.session_state.show_summary)
            setattr(user, "voice_input", st.session_state.voice_input)
            setattr(user, "speak_summary", st.session_state.speak_summary)
            setattr(user, "show_suggested", st.session_state.show_suggested)
            setattr(user, "show_followup", st.session_state.show_followup)
            setattr(user, "show_elapsed_time", st.session_state.show_elapsed_time)
            setattr(user, "llm_fallback", st.session_state.llm_fallback)
            setattr(user, "min_message_id", st.session_state.min_message_id)
            setattr(user, "selected_llm_provider", st.session_state.get("selected_llm_provider"))
            setattr(user, "selected_llm_model", st.session_state.get("selected_llm_model"))

            # Commit the changes to the database
            session.commit()

            st.toast("User data updated successfully!")
        else:
            st.error("User not found.")

        # Close the session
        session.close()
    except Exception as e:
        st.error(f"Error saving user settings: {e}")
        logger.error(f"Error saving user settings: {e}")


def create_user(username: str, password: str, first_name: str, last_name: str, role_id: int, theme: str = None) -> bool:
    """
    Create a new user with the specified details.

    Args:
        username: The username for the new user
        password: The password (will be hashed)
        first_name: User's first name
        last_name: User's last name
        role_id: The ID of the user role

    Returns:
        bool: True if user was created successfully, False otherwise
    """
    try:
        with SessionLocal() as session:
            # Check if username already exists
            existing_user = session.query(User).filter(func.lower(User.username) == username.lower()).first()
            if existing_user:
                return False

            # Hash the password using SHA-256 (matching existing pattern)
            hashed_password = hashlib.sha256(password.encode()).hexdigest()

            # Create new user
            new_user = User(
                username=username,
                password=hashed_password,
                first_name=first_name,
                last_name=last_name,
                user_role_id=role_id,
                show_sql=True,
                show_table=True,
                show_plotly_code=False,
                show_chart=False,
                show_question_history=True,
                show_summary=True,
                voice_input=False,
                speak_summary=False,
                show_suggested=False,
                show_followup=False,
                show_elapsed_time=True,
                llm_fallback=False,
                min_message_id=0,
                theme=theme,
            )

            session.add(new_user)
            session.commit()

            # Log admin action
            try:
                from orm.logging_functions import log_admin_action
                from orm.models import AdminActionType

                admin_id = _get_current_user_id()
                if admin_id:
                    # Refresh to get the new user's ID
                    session.refresh(new_user)
                    log_admin_action(
                        admin_id=admin_id,
                        action_type=AdminActionType.USER_CREATE,
                        description=f"Created user '{username}'",
                        target_user_id=new_user.id,
                        target_username=username,
                        target_entity_type="user",
                    )
            except Exception as e:
                logger.warning("Failed to log user creation for %s: %s", username, e)

            return True

    except Exception as e:
        logger.error(f"Error creating user: {e}")
        return False


def get_all_user_roles():
    """Get all available user roles."""
    try:
        with SessionLocal() as session:
            roles = session.query(UserRole).all()
            return [(role.id, role.role_name, role.description) for role in roles]
    except Exception as e:
        logger.error(f"Error fetching user roles: {e}")
        return []


def get_all_users():
    """Get all users with their roles."""
    try:
        with SessionLocal() as session:
            users = session.query(User).options(joinedload(User.role)).all()
            user_list = []
            for user in users:
                user_list.append(
                    {
                        "id": user.id,
                        "username": user.username,
                        "first_name": user.first_name,
                        "last_name": user.last_name,
                        "role_id": user.user_role_id,
                        "role_name": user.role.role_name if user.role else "No Role",
                        "theme": user.theme,
                        "created_at": user.created_at,
                    }
                )
            return user_list
    except Exception as e:
        logger.error(f"Error fetching users: {e}")
        return []


def update_user(
    user_id: int,
    username: str = None,
    first_name: str = None,
    last_name: str = None,
    role_id: int = None,
    theme: str = None,
) -> bool:
    """Update user details."""
    try:
        with SessionLocal() as session:
            user = session.query(User).filter(User.id == user_id).first()
            if not user:
                return False

            # Capture old values for logging
            old_values = {
                "username": user.username,
                "first_name": user.first_name,
                "last_name": user.last_name,
                "role_id": user.user_role_id,
                "theme": user.theme,
            }

            # Check if new username is taken (if changing username)
            if username and username != user.username:
                existing = (
                    session.query(User)
                    .filter(func.lower(User.username) == username.lower(), User.id != user_id)
                    .first()
                )
                if existing:
                    return False

            # Update fields if provided
            if username:
                user.username = username
            if first_name:
                user.first_name = first_name
            if last_name:
                user.last_name = last_name
            if role_id:
                user.user_role_id = role_id
            if theme:
                user.theme = theme

            session.commit()

            # Log admin action
            try:
                from orm.logging_functions import log_admin_action
                from orm.models import AdminActionType

                admin_id = _get_current_user_id()
                if admin_id:
                    new_values = {
                        "username": username,
                        "first_name": first_name,
                        "last_name": last_name,
                        "role_id": role_id,
                        "theme": theme,
                    }
                    # Only include changed values
                    changes = {k: v for k, v in new_values.items() if v is not None}
                    log_admin_action(
                        admin_id=admin_id,
                        action_type=AdminActionType.USER_UPDATE,
                        description=f"Updated user '{user.username}'",
                        target_user_id=user_id,
                        target_username=user.username,
                        target_entity_type="user",
                        old_value={k: old_values[k] for k in changes.keys()},
                        new_value=changes,
                    )
            except Exception as e:
                logger.warning("Failed to log user update for user_id %s: %s", user_id, e)

            return True

    except Exception as e:
        logger.error(f"Error updating user: {e}")
        return False


def delete_user(user_id: int) -> bool:
    """Delete a user from the system."""
    try:
        with SessionLocal() as session:
            user = session.query(User).filter(User.id == user_id).first()
            if not user:
                return False

            # Capture username before deletion for logging
            deleted_username = user.username

            # Delete associated messages first
            session.query(Message).filter(Message.user_id == user_id).delete()

            # Delete the user
            session.delete(user)
            session.commit()

            # Log admin action
            try:
                from orm.logging_functions import log_admin_action
                from orm.models import AdminActionType

                admin_id = _get_current_user_id()
                if admin_id:
                    log_admin_action(
                        admin_id=admin_id,
                        action_type=AdminActionType.USER_DELETE,
                        description=f"Deleted user '{deleted_username}'",
                        target_user_id=user_id,
                        target_username=deleted_username,
                        target_entity_type="user",
                    )
            except Exception as e:
                logger.warning("Failed to log user deletion for %s: %s", deleted_username, e)

            return True

    except Exception as e:
        logger.error(f"Error deleting user: {e}")
        return False


def get_user(user_id):
    try:
        # Create a new database session
        with SessionLocal() as session:
            # Query to get the user by ID, eagerly loading the role
            user = session.query(User).options(joinedload(User.role)).filter(User.id == user_id).one_or_none()
            return user
    except Exception as e:
        st.error(f"Error getting user: {e}")
        logger.error(f"Error getting user: {e}")


def get_recent_messages():
    try:
        max_index = st.session_state.min_message_id

        user_id = st.session_state.cookies.get("user_id")

        # Create a new database session
        with SessionLocal() as session:
            # Query to get the last 20 messages for the user, excluding those with an index greater than max_index
            messages = (
                session.query(Message)
                .filter(Message.user_id == user_id, Message.id > max_index)
                .order_by(Message.created_at.desc(), Message.id.desc())
                .limit(20)
                .all()
            )

        messages.reverse()

        return messages
    except Exception as e:
        st.error(f"Error getting recent messages: {e}")
        logger.error(f"Error getting recent messages: {e}")


def delete_all_messages():
    try:
        user_id = st.session_state.cookies.get("user_id")

        # Create a new database session
        with SessionLocal() as session:
            session.query(Message).filter(Message.user_id == user_id).delete()
            session.commit()

        st.toast("All messages deleted successfully!")

        st.session_state.messages = []
    except Exception as e:
        st.error(f"Error deleting all messages: {e}")
        logger.error(f"Error deleting all messages: {e}")


def update_user_preferences(user_id: int, **preferences) -> bool:
    """Update boolean user preference flags for a specific user (admin capable)."""
    try:
        with SessionLocal() as session:
            user = session.query(User).filter(User.id == user_id).first()
            if not user:
                return False

            allowed_fields = {
                "show_sql",
                "show_table",
                "show_plotly_code",
                "show_chart",
                "show_question_history",
                "show_summary",
                "voice_input",
                "speak_summary",
                "show_suggested",
                "show_followup",
                "show_elapsed_time",
                "llm_fallback",
                "min_message_id",
                "selected_llm_provider",
                "selected_llm_model",
            }
            for key, value in preferences.items():
                if key in allowed_fields:
                    setattr(user, key, value)

            session.commit()
            return True
    except Exception as e:
        logger.error(f"Error updating user preferences: {e}")
        return False


def admin_change_password(user_id: int, new_password: str) -> bool:
    """Admin-only password set without requiring current password."""
    try:
        with SessionLocal() as session:
            user = session.query(User).filter(User.id == user_id).first()
            if not user:
                return False
            user.password = hashlib.sha256(new_password.encode()).hexdigest()
            session.commit()

            # Log admin action
            try:
                from orm.logging_functions import log_admin_action
                from orm.models import AdminActionType

                admin_id = _get_current_user_id()
                if admin_id:
                    log_admin_action(
                        admin_id=admin_id,
                        action_type=AdminActionType.USER_PASSWORD_RESET,
                        description=f"Reset password for user '{user.username}'",
                        target_user_id=user_id,
                        target_username=user.username,
                        target_entity_type="user",
                    )
            except Exception as e:
                logger.warning("Failed to log admin password reset for user_id %s: %s", user_id, e)

            return True
    except Exception as e:
        logger.error(f"Error setting user password: {e}")
        return False


def get_user_stats_for_all_users():
    """Return a dict of user_id -> stats for questions, charts, errors, dataframes, summaries."""
    try:
        chart_types = [
            MessageType.PLOTLY_CHART.value,
            MessageType.ST_LINE_CHART.value,
            MessageType.ST_BAR_CHART.value,
            MessageType.ST_AREA_CHART.value,
            MessageType.ST_SCATTER_CHART.value,
        ]

        with SessionLocal() as session:
            rows = (
                session.query(
                    Message.user_id,
                    func.sum(case((Message.role == RoleType.USER.value, 1), else_=0)).label("questions"),
                    func.sum(case((Message.type.in_(chart_types), 1), else_=0)).label("charts"),
                    func.sum(case((Message.type == MessageType.ERROR.value, 1), else_=0)).label("errors"),
                    func.sum(case((Message.type == MessageType.DATAFRAME.value, 1), else_=0)).label("dataframes"),
                    func.sum(case((Message.type == MessageType.SUMMARY.value, 1), else_=0)).label("summaries"),
                )
                .group_by(Message.user_id)
                .all()
            )

            stats_map = {}
            for row in rows:
                stats_map[row.user_id] = {
                    "questions": int(row.questions or 0),
                    "charts": int(row.charts or 0),
                    "errors": int(row.errors or 0),
                    "dataframes": int(row.dataframes or 0),
                    "summaries": int(row.summaries or 0),
                }
            return stats_map
    except Exception as e:
        logger.error(f"Error fetching user stats: {e}")
        return {}


def get_user_stats(user_id: int):
    """Return stats dict for a single user."""
    try:
        stats_map = get_user_stats_for_all_users()
        return stats_map.get(
            user_id,
            {"questions": 0, "charts": 0, "errors": 0, "dataframes": 0, "summaries": 0},
        )
    except Exception:
        return {"questions": 0, "charts": 0, "errors": 0, "dataframes": 0, "summaries": 0}


def get_user_daily_stats(user_id: int, days: int = 7):
    """Return a list of dicts with daily counts for the past N days for one user.
    Keys: date, questions, charts, errors, dataframes, summaries
    """
    try:
        chart_types = [
            MessageType.PLOTLY_CHART.value,
            MessageType.ST_LINE_CHART.value,
            MessageType.ST_BAR_CHART.value,
            MessageType.ST_AREA_CHART.value,
            MessageType.ST_SCATTER_CHART.value,
        ]

        with SessionLocal() as session:
            # SQLite strftime('%Y-%m-%d', created_at) groups by date; works also in Postgres with date_trunc but we use generic
            date_expr = func.strftime("%Y-%m-%d", Message.created_at)
            rows = (
                session.query(
                    date_expr.label("d"),
                    func.sum(case((Message.role == RoleType.USER.value, 1), else_=0)).label("questions"),
                    func.sum(case((Message.type.in_(chart_types), 1), else_=0)).label("charts"),
                    func.sum(case((Message.type == MessageType.ERROR.value, 1), else_=0)).label("errors"),
                    func.sum(case((Message.type == MessageType.DATAFRAME.value, 1), else_=0)).label("dataframes"),
                    func.sum(case((Message.type == MessageType.SUMMARY.value, 1), else_=0)).label("summaries"),
                )
                .filter(Message.user_id == user_id)
                .group_by("d")
                .order_by("d")
                .all()
            )

        # Build a dense sequence of dates over the requested window, default missing to zeros
        import datetime as _dt

        today = _dt.date.today()
        start = today - _dt.timedelta(days=days - 1)
        by_date = {r.d: r for r in rows}
        output = []
        for i in range(days):
            d = start + _dt.timedelta(days=i)
            key = d.strftime("%Y-%m-%d")
            r = by_date.get(key)
            output.append(
                {
                    "date": key,
                    "questions": int((r.questions if r else 0) or 0),
                    "charts": int((r.charts if r else 0) or 0),
                    "errors": int((r.errors if r else 0) or 0),
                    "dataframes": int((r.dataframes if r else 0) or 0),
                    "summaries": int((r.summaries if r else 0) or 0),
                }
            )
        return output
    except Exception as e:
        logger.error(f"Error fetching user daily stats: {e}")
        return []


def get_user_recent_questions(user_id: int, limit: int = 200):
    """Return a list of recent question strings asked by the user, newest first.
    Combines user messages' content and assistant messages' question field, de-duplicated by text.
    """
    try:
        with SessionLocal() as session:
            user_q = session.query(Message.content.label("q"), Message.created_at.label("ts")).filter(
                Message.user_id == user_id,
                Message.role == RoleType.USER.value,
                Message.content.isnot(None),
                func.length(Message.content) > 0,
            )
            asst_q = session.query(Message.question.label("q"), Message.created_at.label("ts")).filter(
                Message.user_id == user_id,
                Message.role == RoleType.ASSISTANT.value,
                Message.question.isnot(None),
                func.length(Message.question) > 0,
            )

            union_q = user_q.union_all(asst_q).subquery()
            grouped = (
                session.query(union_q.c.q, func.max(union_q.c.ts).label("ts"))
                .group_by(union_q.c.q)
                .order_by(func.max(union_q.c.ts).desc())
                .limit(limit)
                .all()
            )
            return [r.q for r in grouped]
    except Exception as e:
        logger.error(f"Error fetching user recent questions: {e}")
        return []


def get_user_questions_page(user_id: int, page: int = 1, page_size: int = 50):
    """Paginated questions: returns dict with items [{question, created_at, status, elapsed_seconds}], total count.
    Combines user message content and assistant question field, deduped by question text; shows the most recent timestamp.
    Success is true if any assistant message for that question produced a non-error result (dataframe/summary/chart).
    """
    try:
        with SessionLocal() as session:
            # Combine user content and assistant question, then dedupe by text keeping most recent timestamp
            user_q = session.query(Message.content.label("q"), Message.created_at.label("ts")).filter(
                Message.user_id == user_id,
                Message.role == RoleType.USER.value,
                Message.content.isnot(None),
                func.length(Message.content) > 0,
            )
            asst_q = session.query(Message.question.label("q"), Message.created_at.label("ts")).filter(
                Message.user_id == user_id,
                Message.role == RoleType.ASSISTANT.value,
                Message.question.isnot(None),
                func.length(Message.question) > 0,
            )
            union_q = user_q.union_all(asst_q).subquery()
            grouped_sub = (
                session.query(union_q.c.q.label("q"), func.max(union_q.c.ts).label("created_at"))
                .group_by(union_q.c.q)
                .subquery()
            )

            total = session.query(func.count()).select_from(grouped_sub).scalar() or 0

            rows = (
                session.query(grouped_sub.c.q, grouped_sub.c.created_at)
                .order_by(grouped_sub.c.created_at.desc())
                .offset(max(0, (page - 1) * page_size))
                .limit(page_size)
                .all()
            )

            questions = [r[0] for r in rows]

            # If no rows, return early
            if not rows:
                return {"items": [], "total": int(total)}

            # Aggregate assistant results by question across the fetched set
            chart_types = [
                MessageType.PLOTLY_CHART.value,
                MessageType.ST_LINE_CHART.value,
                MessageType.ST_BAR_CHART.value,
                MessageType.ST_AREA_CHART.value,
                MessageType.ST_SCATTER_CHART.value,
            ]

            # Use assistant-side Message.question to match; include user content too for robustness
            agg = (
                session.query(
                    Message.question.label("q"),
                    func.sum(case((Message.type.in_(chart_types), 1), else_=0)).label("charts"),
                    func.sum(case((Message.type == MessageType.DATAFRAME.value, 1), else_=0)).label("dataframes"),
                    func.sum(case((Message.type == MessageType.SUMMARY.value, 1), else_=0)).label("summaries"),
                    func.sum(case((Message.type == MessageType.ERROR.value, 1), else_=0)).label("errors"),
                    func.sum(func.coalesce(Message.elapsed_time, 0)).label("elapsed"),
                )
                .filter(
                    Message.user_id == user_id,
                    Message.role == RoleType.ASSISTANT.value,
                    Message.question.isnot(None),
                    Message.question.in_(questions),
                )
                .group_by(Message.question)
                .all()
            )
            metrics = {row.q: row for row in agg}

            items = []
            for q, created_at in rows:
                m = metrics.get(q)
                charts = int(getattr(m, "charts", 0) or 0) if m else 0
                dataframes = int(getattr(m, "dataframes", 0) or 0) if m else 0
                summaries = int(getattr(m, "summaries", 0) or 0) if m else 0
                errors = int(getattr(m, "errors", 0) or 0) if m else 0
                elapsed = float(getattr(m, "elapsed", 0) or 0.0) if m else 0.0
                success = (charts + dataframes + summaries) > 0 and errors == 0
                items.append(
                    {
                        "question": q,
                        "created_at": created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at),
                        "status": "Success" if success else ("Error" if errors > 0 else "Unknown"),
                        "elapsed_seconds": round(elapsed, 6),
                    }
                )

            return {"items": items, "total": int(total)}
    except Exception as e:
        logger.error(f"Error fetching user questions page: {e}")
        return {"items": [], "total": 0}
