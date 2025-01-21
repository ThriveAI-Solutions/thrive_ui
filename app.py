import streamlit as st

# --- PAGE SETUP ---
login_page = st.Page(
    page="views/login.py",
    title="Login",
    icon="🔓",
    default=True
)
chat_bot_page = st.Page(
    page = "views/chat_bot.py",
    title = "Chat Bot",
    icon = "🤖"
)

pg = st.navigation(pages=[login_page, chat_bot_page])

if st.sidebar.button("Logout", use_container_width=True, type="primary"):
    st.session_state["logged_in"] = False #TODO: update this in the cookies
    st.switch_page("views/login.py")

st.write("Authenticated:", st.session_state.get("logged_in"))

pg.run()