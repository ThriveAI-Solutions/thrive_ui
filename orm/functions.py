import hashlib
import json
import re

import streamlit as st
from sqlalchemy import case, func
from sqlalchemy.orm import joinedload

from orm.models import Message, SessionLocal, User, UserRole
from utils.enums import MessageType, RoleType
from utils.quick_logger import get_logger

# Lightweight email format check per Epic #98 — no MX lookups, no RFC compliance.
# Requires at least one non-space + "@" + non-space + "." + non-space.
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _is_valid_email(value: str | None) -> bool:
    if value is None:
        return False
    stripped = value.strip()
    if not stripped:
        return False
    return bool(_EMAIL_RE.match(stripped))


logger = get_logger(__name__)


# ── Required-field validation (Epic #179) ─────────────────────────────────
#
# Server-authoritative validation for the three columns that became NOT NULL
# in Alembic revision 7b3a1f0c92d4: ``email``, ``organization``,
# ``user_role_id``. Both ``create_user`` and ``update_user`` raise
# :class:`UserValidationError` *before* opening a DB session when any of these
# fields are missing/empty/malformed. Other failure modes (duplicate
# username, duplicate email, DB error) still return ``False`` so existing
# callers that only branch on the boolean don't need a try/except.
#
# Field names in ``missing_fields`` use the public API names callers see in
# the dialog forms — ``email``, ``organization``, ``role`` — so the UI can
# map each entry directly to the offending input. ``user_role_id`` is
# reported as ``"role"`` since that's the label on the dialog.


class UserValidationError(ValueError):
    """Raised when create_user / update_user receive missing or invalid required fields.

    ``missing_fields`` is the ordered list of field names that failed
    validation (e.g. ``["email", "organization", "role"]``). The UI maps
    each entry to a per-field error message.
    """

    def __init__(self, missing_fields: list[str], message: str | None = None) -> None:
        self.missing_fields = list(missing_fields)
        if message is None:
            message = f"User validation failed: missing or invalid {', '.join(self.missing_fields)}"
        super().__init__(message)


def _validate_required_user_fields(
    *,
    email: str | None,
    organization: str | None,
    role_id: int | None,
    require_role_exists: bool = True,
) -> None:
    """Raise UserValidationError when any required field is missing/invalid.

    ``email``  — must be non-empty after strip and match the lightweight regex.
    ``organization`` — must be non-empty after strip.
    ``role_id`` — must be a positive int and (when ``require_role_exists``)
                  point to an existing ``thrive_user_role`` row.

    Called by ``create_user`` (with all three values required) and by
    ``update_user`` (only checks values the caller is actually changing —
    None means "don't touch").
    """
    missing: list[str] = []

    # Email
    if not _is_valid_email(email):
        missing.append("email")

    # Organization
    if organization is None or not organization.strip():
        missing.append("organization")

    # Role
    if role_id is None or not isinstance(role_id, int) or isinstance(role_id, bool) or role_id <= 0:
        missing.append("role")
    elif require_role_exists:
        # Cheap FK check — keeps a bad role_id from getting through and
        # cascading into a less actionable IntegrityError downstream.
        try:
            with SessionLocal() as session:
                exists = session.query(UserRole.id).filter(UserRole.id == role_id).first()
            if exists is None:
                missing.append("role")
        except Exception as exc:
            # If the role table is unreachable we can't verify the FK; log
            # and treat it as a missing role so the caller sees the error
            # rather than a downstream IntegrityError.
            logger.warning("Could not verify role_id=%s exists: %s", role_id, exc)
            missing.append("role")

    if missing:
        raise UserValidationError(missing)


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

        # Cookie points to a user_id that no longer exists (e.g. the SQLite DB
        # was recreated/migrated and IDs shifted). Treat as not-loaded so the
        # caller can clear the stale cookie and re-prompt for login, instead of
        # cascading into a series of 'NoneType' attribute errors.
        if user is None:
            logger.warning("Cookie user_id %s does not resolve to a user; treating as logged out.", user_id)
            return None

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
        st.session_state.show_thinking_process = bool(getattr(user, "show_thinking_process", False))
        st.session_state.llm_fallback = user.llm_fallback
        st.session_state.confirm_magic_commands = getattr(user, "confirm_magic_commands", True)
        st.session_state.show_community_engagement = bool(getattr(user, "show_community_engagement", False) or False)
        agentic_pref = getattr(user, "agentic_mode", True)
        st.session_state.agentic_mode = bool(agentic_pref) if agentic_pref is not None else True
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
            setattr(user, "show_thinking_process", st.session_state.get("show_thinking_process", False))
            setattr(user, "llm_fallback", st.session_state.llm_fallback)
            setattr(user, "confirm_magic_commands", st.session_state.get("confirm_magic_commands", True))
            setattr(user, "show_community_engagement", st.session_state.get("show_community_engagement", False))
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


def create_user(
    username: str,
    password: str,
    first_name: str,
    last_name: str,
    role_id: int,
    *,
    email: str,
    organization: str,
    theme: str | None = None,
) -> bool:
    """
    Create a new user with the specified details.

    Args:
        username: The username for the new user
        password: The password (will be hashed)
        first_name: User's first name
        last_name: User's last name
        role_id: The ID of the user role
        email: Required keyword-only. Validated against a lightweight regex
            and rejected if a case-insensitive duplicate already exists.
        organization: Required keyword-only. Stripped and rejected if empty.
        theme: Optional theme preference.

    Returns:
        bool: True if the user was created. ``False`` for non-validation
        failures (duplicate username, duplicate email, DB error) so
        existing callers that branch on bool don't need a try/except.

    Raises:
        UserValidationError: When ``email``, ``organization``, or
        ``role_id`` are missing/empty/malformed. Carries the list of
        offending field names via ``missing_fields`` so the UI can render
        per-field error messages. Epic #179.
    """
    # Trim surrounding whitespace so a stray space (e.g. typed into the
    # admin create-user form) can't silently break login — credential
    # checks match on the exact stored username. Password is intentionally
    # left untouched.
    username = (username or "").strip()
    first_name = (first_name or "").strip()
    last_name = (last_name or "").strip()
    email = (email or "").strip()
    organization = (organization or "").strip()

    # Server-authoritative required-field validation (Epic #179) — runs
    # *before* opening a session. UserValidationError carries the list of
    # missing fields so UI callers can render per-field error messages.
    _validate_required_user_fields(email=email, organization=organization, role_id=role_id)

    try:
        with SessionLocal() as session:
            # Check if username already exists
            existing_user = session.query(User).filter(func.lower(User.username) == username.lower()).first()
            if existing_user:
                return False

            # Check if email already exists (case-insensitive).
            existing_email = session.query(User).filter(func.lower(User.email) == email.lower()).first()
            if existing_email:
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
                email=email,
                organization=organization,
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
                        "email": user.email,
                        "organization": user.organization,
                        "created_at": user.created_at,
                    }
                )
            return user_list
    except Exception as e:
        logger.error(f"Error fetching users: {e}")
        return []


def _is_admin_db(admin_id: int) -> bool:
    """Defense-in-depth role check: re-resolve the caller's role from the DB.

    Does not consult ``st.session_state``. Returns True only when the row
    exists and resolves to ``RoleTypeEnum.ADMIN``.
    """
    from orm.models import RoleTypeEnum

    try:
        with SessionLocal() as session:
            row = session.query(User).options(joinedload(User.role)).filter(User.id == admin_id).one_or_none()
            if row is None or row.role is None:
                return False
            return row.role.role == RoleTypeEnum.ADMIN
    except Exception as e:
        logger.warning("Failed to verify admin role for user_id=%s: %s", admin_id, e)
        return False


def get_users_for_export(admin_id: int) -> tuple[list[dict], int]:
    """Return the user roster in Bulk-Import-compatible CSV row shape.

    Columns (in order): ``UserID, First Name, Last Name, Email, Organization, Role``.
    Excludes ``password``, ``okta_sub``, and all preference fields.

    Enforces an admin-only gate independently of UI state (defense in depth)
    and writes a single ``thrive_admin_action`` row per call — ``success=True``
    on the happy path, ``success=False`` with a rejection ``error_message`` when
    the caller is not an admin.
    """
    from orm.logging_functions import log_admin_action
    from orm.models import AdminActionType

    if not _is_admin_db(admin_id):
        log_admin_action(
            admin_id=admin_id,
            action_type=AdminActionType.USER_EXPORT,
            description="Non-admin attempted user export",
            target_entity_type="user",
            success=False,
            error_message="caller is not an admin",
        )
        return [], 0

    try:
        with SessionLocal() as session:
            users = session.query(User).options(joinedload(User.role)).all()
            rows = [
                {
                    "UserID": u.username,
                    "First Name": u.first_name,
                    "Last Name": u.last_name,
                    "Email": u.email,
                    "Organization": u.organization,
                    "Role": u.role.role_name if u.role else "No Role",
                }
                for u in users
            ]
        n = len(rows)
        log_admin_action(
            admin_id=admin_id,
            action_type=AdminActionType.USER_EXPORT,
            description=f"Exported {n} users to CSV",
            target_entity_type="user",
            affected_count=n,
            success=True,
        )
        return rows, n
    except Exception as e:
        logger.error("Error exporting users: %s", e)
        log_admin_action(
            admin_id=admin_id,
            action_type=AdminActionType.USER_EXPORT,
            description="User export failed",
            target_entity_type="user",
            success=False,
            error_message=str(e),
        )
        return [], 0


def update_user(
    user_id: int,
    username: str = None,
    first_name: str = None,
    last_name: str = None,
    role_id: int = None,
    theme: str = None,
    *,
    email: str | None = None,
    organization: str | None = None,
) -> bool:
    """Update user details.

    Only fields with non-None kwargs are touched (legacy semantics
    preserved). When a caller *does* pass a value for one of the three
    required fields (``email``, ``organization``, ``role_id``), it must
    pass validation — clearing one of them to an empty/invalid value
    raises :class:`UserValidationError` instead of writing NULL/garbage.

    Returns:
        ``True`` on success, ``False`` for non-validation failures
        (user not found, duplicate username/email collision, DB error).

    Raises:
        UserValidationError: When the caller passes an empty/invalid
        value for one of the required fields. ``missing_fields`` carries
        the offending field name(s). Epic #179.
    """
    # Strip the optional new fields up front so the validator and the
    # downstream DB writes see the same canonical values.
    if email is not None:
        email = email.strip()
    if organization is not None:
        organization = organization.strip()

    # Validate any required field the caller is actually trying to
    # update. None means "don't touch" — only validate explicit values.
    fields_to_check = {}
    if email is not None:
        fields_to_check["email"] = email
    if organization is not None:
        fields_to_check["organization"] = organization
    if role_id is not None:
        fields_to_check["role_id"] = role_id

    if fields_to_check:
        # Only check fields the caller passed; missing-from-update fields
        # are explicitly not validated (legacy "None means don't touch").
        # We do the per-field check inline rather than calling the shared
        # validator because the shared validator requires all three.
        missing: list[str] = []
        if "email" in fields_to_check and not _is_valid_email(fields_to_check["email"]):
            missing.append("email")
        if "organization" in fields_to_check and not fields_to_check["organization"]:
            missing.append("organization")
        if "role_id" in fields_to_check:
            rid = fields_to_check["role_id"]
            if not isinstance(rid, int) or isinstance(rid, bool) or rid <= 0:
                missing.append("role")
            else:
                try:
                    with SessionLocal() as session:
                        exists = session.query(UserRole.id).filter(UserRole.id == rid).first()
                    if exists is None:
                        missing.append("role")
                except Exception as exc:
                    logger.warning("Could not verify role_id=%s exists: %s", rid, exc)
                    missing.append("role")
        if missing:
            raise UserValidationError(missing)

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
                "email": user.email,
                "organization": user.organization,
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

            # Check if new email is taken (if changing email, case-insensitive)
            if email is not None and (user.email is None or email.lower() != user.email.lower()):
                existing_email = (
                    session.query(User).filter(func.lower(User.email) == email.lower(), User.id != user_id).first()
                )
                if existing_email:
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
            if email is not None:
                user.email = email
            if organization is not None:
                user.organization = organization

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
                        "email": email,
                        "organization": organization,
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
                "show_community_engagement",
                "min_message_id",
                "selected_llm_provider",
                "selected_llm_model",
                "agentic_mode",
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
