import streamlit as st

# --- PAGE SETUP ---
login_page = st.Page(
    page="views/login.py",
    title="Log In - Thrive AI",
    icon="🔓",
    default=True
)
chat_bot_page = st.Page(
    page = "views/chat_bot.py",
    title = "Chat Bot - Thrive AI",
    icon = "🤖"
)

pg = st.navigation(pages=[login_page, chat_bot_page])

pg.run()