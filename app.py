import streamlit as st
from streamlit_cookies_manager import EncryptedCookieManager
from helperClasses.auth import (check_authenticate)


# TODO: **Update streamlit_cookies_manager**: The behavior of `st.cache` was updated in Streamlit 1.36 to the new caching
# logic used by `st.cache_data` and `st.cache_resource`. This might lead to some problems
# or unexpected behavior in certain edge cases.

# Initialize the cookie manager
st.session_state.cookies = EncryptedCookieManager(prefix="thrive_ai_", password=st.secrets["cookie"]["password"])

# Ensure the cookie manager is initialized
if not st.session_state.cookies.ready():
    st.stop()

# --- PAGE SETUP ---
chat_bot_page = st.Page(
    page = "views/chat_bot.py",
    title = "Chat Bot - Thrive AI",
    icon = "🤖"
)

pg = st.navigation(pages=[chat_bot_page])

check_authenticate()

pg.run()