import streamlit as st
print('app')
def logout_callback(arg):
    st.switch_page("views/login.py")

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

pg.run()

if 'authentication_status' not in st.session_state:
    print('goto login')
    st.switch_page('views/login.py')

if "authenticator" in st.session_state:
    print('authenticator')
    authenticator = st.session_state["authenticator"]

    if st.session_state["authentication_status"]:
        authenticator.logout(location='sidebar',  callback=logout_callback)