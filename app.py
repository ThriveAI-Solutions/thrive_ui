import streamlit as st
from streamlit_cookies_manager import EncryptedCookieManager

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

pg.run()