import hashlib
import json
import logging

import streamlit as st
from sqlalchemy import func
from sqlalchemy.orm import joinedload

from orm.models import Message, SessionLocal, User, UserRole

logger = logging.getLogger(__name__)


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


def create_user(username: str, password: str, first_name: str, last_name: str, role_id: int) -> bool:
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
                min_message_id=0
            )
            
            session.add(new_user)
            session.commit()
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
                user_list.append({
                    'id': user.id,
                    'username': user.username,
                    'first_name': user.first_name,
                    'last_name': user.last_name,
                    'role_id': user.user_role_id,
                    'role_name': user.role.role_name if user.role else 'No Role',
                    'created_at': user.created_at
                })
            return user_list
    except Exception as e:
        logger.error(f"Error fetching users: {e}")
        return []


def update_user(user_id: int, username: str = None, first_name: str = None, 
                last_name: str = None, role_id: int = None) -> bool:
    """Update user details."""
    try:
        with SessionLocal() as session:
            user = session.query(User).filter(User.id == user_id).first()
            if not user:
                return False
            
            # Check if new username is taken (if changing username)
            if username and username != user.username:
                existing = session.query(User).filter(
                    func.lower(User.username) == username.lower(),
                    User.id != user_id
                ).first()
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
            
            session.commit()
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
            
            # Delete associated messages first
            session.query(Message).filter(Message.user_id == user_id).delete()
            
            # Delete the user
            session.delete(user)
            session.commit()
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
