import logging
import secrets
from datetime import datetime, timedelta
import utils.chat_bot_helper as chat_helper # need to do this even though its not used,  so that the class is initialized before its used.
import streamlit as st
from orm.functions import set_user_preferences_in_session_state, verify_user_credentials

logger = logging.getLogger(__name__)

# CSRF token storage
csrf_tokens = {}


def generate_csrf_token(user_id: str) -> str:
    """Generate a CSRF token for the user session."""
    token = secrets.token_urlsafe(32)
    csrf_tokens[user_id] = {
        'token': token,
        'created_at': datetime.now()
    }
    return token


def validate_csrf_token(user_id: str, token: str) -> bool:
    """Validate a CSRF token."""
    if user_id not in csrf_tokens:
        return False
    
    stored_token_data = csrf_tokens[user_id]
    
    # Check if token is expired (valid for 1 hour)
    if datetime.now() - stored_token_data['created_at'] > timedelta(hours=1):
        del csrf_tokens[user_id]
        return False
    
    return stored_token_data['token'] == token


def secure_logout(user_id: str):
    """Perform secure logout with proper session cleanup."""
    try:
        # Clear CSRF token
        if user_id in csrf_tokens:
            del csrf_tokens[user_id]
        
        # Clear session cookies
        st.session_state.cookies["user_id"] = ""
        st.session_state.cookies["expiry_date"] = ""
        st.session_state.cookies["csrf_token"] = ""
        st.session_state.cookies.save()
        
        # Clear session state
        session_keys_to_clear = ['messages', 'show_sql', 'show_table', 'show_plotly_code', 
                               'show_chart', 'show_question_history', 'show_summary', 
                               'voice_input', 'speak_summary', 'show_suggested', 
                               'show_followup', 'show_elapsed_time', 'llm_fallback', 
                               'min_message_id', 'user_role', 'loaded']
        
        for key in session_keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        
        logger.info(f"Secure logout completed for user: {user_id}")
        
    except Exception as e:
        logger.error(f"Error during secure logout: {e}")


def check_authenticate():
    try:
        user_id = st.session_state.cookies.get("user_id")
        expiry_date_str = st.session_state.cookies.get("expiry_date")
        csrf_token = st.session_state.cookies.get("csrf_token")
        
        if user_id and expiry_date_str:
            expiry_date = datetime.fromisoformat(expiry_date_str)
            
            # Check if session is expired
            if datetime.now() >= expiry_date:
                logger.info(f"Session expired for user: {user_id}")
                secure_logout(user_id)
                show_login()
                return
            
            # Validate CSRF token
            if not csrf_token or not validate_csrf_token(user_id, csrf_token):
                logger.warning(f"Invalid CSRF token for user: {user_id}")
                secure_logout(user_id)
                show_login()
                return
            
            # Set user preferences and display welcome message
            user = set_user_preferences_in_session_state()
            
            if user:
                cols = st.sidebar.columns([0.7, 0.3], vertical_alignment="bottom")
                with cols[0]:
                    st.title(f"Welcome {user.first_name} {user.last_name}")
                with cols[1]:
                    logout = st.button("Log Out")
                    if logout:
                        secure_logout(user_id)
                        st.rerun()
            else:
                logger.error(f"Failed to load user preferences for user: {user_id}")
                secure_logout(user_id)
                show_login()
        else:
            show_login()
    except Exception as e:
        logger.error(f"Error checking authentication: {e}")
        if user_id:
            secure_logout(user_id)
        show_login()


def show_login():
    try:
        # Add security headers
        st.markdown(
            """
        <style>
            [data-testid="stSidebarCollapsedControl"], [data-testid="stSidebar"] {
                display: none
            }
        </style>
        """,
            unsafe_allow_html=True,
        )
        
        # Display security notice
        st.markdown(
            """
            <div style="background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
                <h4>🔒 Security Notice</h4>
                <p>This system implements enhanced security measures including:</p>
                <ul>
                    <li>Rate limiting (5 failed attempts = 15 min lockout)</li>
                    <li>Strong password requirements</li>
                    <li>Secure session management</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        st.image("assets/logo.png", width=200)
        st.title("🔓 Log In")

        with st.form("login_form"):
            username = st.text_input("Username", max_chars=50)
            password = st.text_input("Password", type="password", max_chars=128)
            submit_button = st.form_submit_button("Login", type="primary")

            if submit_button:
                # Basic client-side validation
                if not username or not password:
                    st.error("Please enter both username and password.")
                elif len(username) > 50:
                    st.error("Username too long.")
                elif len(password) > 128:
                    st.error("Password too long.")
                else:
                    # Attempt login
                    if verify_user_credentials(username, password):  # sets user_id in cookies
                        user_id = st.session_state.cookies.get("user_id")
                        
                        # Generate CSRF token
                        csrf_token = generate_csrf_token(user_id)
                        st.session_state.cookies["csrf_token"] = csrf_token
                        
                        # Set secure session expiry (8 hours)
                        expiry_date = datetime.now() + timedelta(hours=8)
                        st.session_state.cookies["expiry_date"] = expiry_date.isoformat()
                        st.session_state.cookies.save()
                        
                        logger.info(f"Successful login for user: {username}")
                        st.success("Login successful! Redirecting...")
                        st.rerun()
                    else:
                        # Error message already handled in verify_user_credentials
                        pass
        
        # Display password requirements
        with st.expander("Password Requirements"):
            st.markdown("""
            **Strong passwords must contain:**
            - At least 8 characters
            - At least one uppercase letter
            - At least one lowercase letter  
            - At least one number
            - At least one special character (!@#$%^&*()_+-=[]{}|;:,.<>?)
            """)
        
        st.stop()
    except Exception as e:
        st.error(f"Error showing login: {e}")
        logger.error(f"Error showing login: {e}")
