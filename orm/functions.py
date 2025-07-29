import hashlib
import json
import logging
import secrets
import time
from datetime import datetime, timedelta

import streamlit as st
from sqlalchemy import func
from sqlalchemy.orm import joinedload

from orm.models import Message, SessionLocal, User, UserRole

logger = logging.getLogger(__name__)

# Dictionary to track login attempts: {username: {'count': int, 'last_attempt': datetime}}
login_attempts = {}


def hash_password(password: str, salt: bytes = None) -> tuple[str, bytes]:
    """
    Hash a password using PBKDF2 with SHA-256 and a random salt.
    Returns (hashed_password, salt)
    """
    if salt is None:
        salt = secrets.token_bytes(32)
    
    # Use PBKDF2 with 100,000 iterations for strong password hashing
    hashed = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
    return hashed.hex(), salt


def verify_password(password: str, hashed_password: str, salt: bytes) -> bool:
    """
    Verify a password against its hash and salt.
    """
    try:
        test_hash, _ = hash_password(password, salt)
        return test_hash == hashed_password
    except Exception as e:
        logger.error(f"Error verifying password: {e}")
        return False


def is_rate_limited(username: str) -> bool:
    """
    Check if a user is rate limited based on failed login attempts.
    """
    if username not in login_attempts:
        return False
    
    attempt_data = login_attempts[username]
    
    # If more than 5 failed attempts
    if attempt_data['count'] >= 5:
        # Check if lockout period (15 minutes) has passed
        time_since_last_attempt = datetime.now() - attempt_data['last_attempt']
        if time_since_last_attempt < timedelta(minutes=15):
            return True
        else:
            # Reset counter after lockout period
            login_attempts[username] = {'count': 0, 'last_attempt': datetime.now()}
    
    return False


def record_failed_login(username: str):
    """
    Record a failed login attempt.
    """
    if username not in login_attempts:
        login_attempts[username] = {'count': 0, 'last_attempt': datetime.now()}
    
    login_attempts[username]['count'] += 1
    login_attempts[username]['last_attempt'] = datetime.now()
    
    logger.warning(f"Failed login attempt for user: {username}. Count: {login_attempts[username]['count']}")


def reset_login_attempts(username: str):
    """
    Reset failed login attempts for a user after successful login.
    """
    if username in login_attempts:
        del login_attempts[username]


def sanitize_input(input_string: str) -> str:
    """
    Basic input sanitization to prevent injection attacks.
    """
    if not input_string:
        return ""
    
    # Remove null bytes and control characters
    sanitized = input_string.replace('\x00', '').replace('\r', '').replace('\n', ' ')
    
    # Limit length to prevent DoS
    return sanitized[:255].strip()


def verify_user_credentials(username: str, password: str) -> bool:
    try:
        # Input validation
        if not username or not password:
            logger.warning("Login attempt with empty username or password")
            return False
        
        # Sanitize inputs
        username = sanitize_input(username)
        password = sanitize_input(password)
        
        # Check rate limiting
        if is_rate_limited(username):
            logger.warning(f"Rate limited login attempt for user: {username}")
            st.error("Too many failed login attempts. Please try again in 15 minutes.")
            return False
        
        # Create a new database session
        with SessionLocal() as session:
            # Query to get the user by username
            user = (
                session.query(User)
                .filter(func.lower(User.username) == username.lower())
                .one_or_none()
            )

            if user:
                # Check if user has new secure hash (has salt) or old SHA-256 hash (no salt)
                if hasattr(user, 'salt') and user.salt:
                    # New secure hash, verify normally
                    if verify_password(password, user.password, user.salt):
                        # Set session data
                        st.session_state.cookies["user_id"] = json.dumps(user.id)
                        userRole = session.query(UserRole).filter(UserRole.id == user.user_role_id).one_or_none()
                        st.session_state.cookies["role_name"] = userRole.role_name
                        
                        reset_login_attempts(username)
                        return True
                elif len(user.password) == 64 and user.password.isalnum():
                    # Old SHA-256 hash, verify and upgrade
                    old_hash = hashlib.sha256(password.encode()).hexdigest()
                    if old_hash == user.password:
                        # Upgrade to new secure hash
                        new_hash, salt = hash_password(password)
                        user.password = new_hash
                        user.salt = salt
                        session.commit()
                        logger.info(f"Upgraded password hash for user: {username}")
                        
                        # Set session data
                        st.session_state.cookies["user_id"] = json.dumps(user.id)
                        userRole = session.query(UserRole).filter(UserRole.id == user.user_role_id).one_or_none()
                        st.session_state.cookies["role_name"] = userRole.role_name
                        
                        reset_login_attempts(username)
                        return True
            
            # Record failed login attempt
            record_failed_login(username)
            return False
            
    except Exception as e:
        logger.error(f"Error verifying user credentials: {e}")
        record_failed_login(username)
        return False


def validate_password_strength(password: str) -> tuple[bool, str]:
    """
    Validate password strength according to security best practices.
    Returns (is_valid, error_message)
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    
    if len(password) > 128:
        return False, "Password must be less than 128 characters long"
    
    has_upper = any(c.isupper() for c in password)
    has_lower = any(c.islower() for c in password)
    has_digit = any(c.isdigit() for c in password)
    has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)
    
    if not (has_upper and has_lower and has_digit and has_special):
        return False, "Password must contain at least one uppercase letter, lowercase letter, digit, and special character"
    
    # Check for common weak passwords
    common_passwords = ["password", "123456", "qwerty", "admin", "letmein", "welcome"]
    if password.lower() in common_passwords:
        return False, "Password is too common. Please choose a stronger password"
    
    return True, ""


def change_password(user_id: int, current_password: str, new_password: str) -> bool:
    try:
        # Validate new password strength
        is_valid, error_msg = validate_password_strength(new_password)
        if not is_valid:
            st.error(error_msg)
            return False
        
        # Sanitize inputs
        current_password = sanitize_input(current_password)
        new_password = sanitize_input(new_password)
        
        # Create a new database session
        with SessionLocal() as session:
            # Retrieve the existing user from the database
            user = session.query(User).filter(User.id == user_id).first()

            if user:
                # Check if current password is correct
                current_password_valid = False
                
                # Check if user has new secure hash (has salt) or old SHA-256 hash (no salt)
                if hasattr(user, 'salt') and user.salt:
                    # New secure hash
                    current_password_valid = verify_password(current_password, user.password, user.salt)
                elif len(user.password) == 64 and user.password.isalnum():
                    # Old SHA-256 hash
                    current_password_hashed = hashlib.sha256(current_password.encode()).hexdigest()
                    current_password_valid = current_password_hashed == user.password
                
                if current_password_valid:
                    # Update the user's password with new secure hash
                    new_hash, salt = hash_password(new_password)
                    user.password = new_hash
                    user.salt = salt
                    session.commit()
                    logger.info(f"Password changed for user ID: {user_id}")
                    return True
                else:
                    st.error("Current password is incorrect")
                    return False
            else:
                st.error("User not found")
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
