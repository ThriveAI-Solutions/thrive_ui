import streamlit as st
import json
import hashlib
from sqlalchemy import func
from orm.models import User, Message, SessionLocal

def verify_user_credentials(username: str, password: str) -> bool:
    # Create a new database session
    session = SessionLocal()

    # Hash the password using SHA-256
    hashed_password = hashlib.sha256(password.encode()).hexdigest()

    # Query to check if the username and hashed password exist in the users table
    user = session.query(User).filter(func.lower(User.username) == username.lower(), User.password == hashed_password).first()
    st.session_state.cookies["user_id"] = json.dumps(user.id)

    # Close the database session
    session.close()

    # Return True if the user exists, otherwise return False
    return user is not None

def set_user_preferences_in_session_state(user):
    if "loaded" not in st.session_state:
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
        st.session_state.llm_fallback = user.llm_fallback
        st.session_state.loaded = True # dont call after initial load
    
def save_user_settings():
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
        setattr(user, "llm_fallback", st.session_state.llm_fallback)
        
        # Commit the changes to the database
        session.commit()

        st.success("User data updated successfully!")
    else:
        st.error("User not found.")

    # Close the session
    session.close()

def get_user(user_id):
    # Create a new database session
    session = SessionLocal()

    # Query to get the user by ID
    user = session.query(User).filter(User.id == user_id).first()

    # Close the session
    session.close()

    return user

def get_recent_messages():
    user_id = st.session_state.cookies.get("user_id")

    # Create a new database session
    session = SessionLocal()

    messages = session.query(Message).filter(Message.user_id == user_id).order_by(Message.created_at.desc()).limit(20).all()

    # Close the session
    session.close()

    messages.reverse()

    return messages