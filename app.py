import logging

import streamlit as st

from utils.discord_logging import add_discord_handler_if_configured, initialize_discord_logging_after_streamlit
from utils.logging_config import setup_logging

# Set the page configuration to wide mode
st.set_page_config(layout="wide")
from streamlit_cookies_manager_ext import EncryptedCookieManager

from utils.auth import check_authenticate

# setup logging
setup_logging(debug=True)

logger = logging.getLogger(__name__)

# Initialize Discord logging after Streamlit is ready

# add_discord_handler_if_configured(logger)

# silence watchdog warnings
logging.getLogger("fsevents").setLevel(logging.INFO)
logging.getLogger("chromadb").setLevel(logging.INFO)
logging.getLogger("httpcore").setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.INFO)

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

# Conditionally add Admin Analytics page for admins
pages = [chat_bot_page, user_page]
if st.session_state.get("user_role") == 0:  # RoleTypeEnum.ADMIN.value = 0
    analytics_page = st.Page(
        page="views/admin_analytics.py",
        title="Admin Analytics",
        icon="📈",
    )
    pages.append(analytics_page)

pg = st.navigation(pages=pages)

check_authenticate()

initialize_discord_logging_after_streamlit(st.session_state.username)

pg.run()
