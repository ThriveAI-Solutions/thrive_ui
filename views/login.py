import streamlit as st
from helperClasses.auth import (is_logged_in)

if is_logged_in() is True:
    st.switch_page("views/chat_bot.py")
    
#--- HIDE SIDEBAR ---
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
#--- HIDE SIDEBAR ---

st.title("🔓 Log In - Thrive AI")

with st.form("login_form"):
    username = st.text_input("Username", value="ThriveAI")
    password = st.text_input("Password", type="password", value="AIThrive")
    submit_button = st.form_submit_button("Login")

    if submit_button:
        if username == "ThriveAI" and password == "AIThrive":
            st.session_state["logged_in"] = True #TODO: update this in the cookies
            st.success("You are now logged in!")
            st.switch_page("views/chat_bot.py")
        else:
            st.error("Incorrect username or password. Please try again.")
