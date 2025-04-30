import streamlit as st

# Set the page configuration to wide mode
st.set_page_config(layout="wide")
from streamlit_cookies_manager_ext import EncryptedCookieManager
from utils.auth import check_authenticate

# Initialize the cookie manager
st.session_state.cookies = EncryptedCookieManager(
    prefix="thrive_ai_",
    password=st.secrets["cookie"]["password"],
)

# Ensure the cookie manager is initialized
if not st.session_state.cookies.ready():
    st.stop()

# --- PAGE SETUP ---
chat_bot_page = st.Page(
    page="views/chat_bot.py",
    title="Chat Bot",
    icon="🤖",
)

user_page = st.Page(
    page="views/user.py",
    title="User Settings",
    icon="👤",
)

pg = st.navigation(pages=[chat_bot_page, user_page])

check_authenticate()

pg.run()
