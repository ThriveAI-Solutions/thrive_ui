import streamlit as st

def check_authenticate():
    if st.session_state.cookies.get("logged_in") == "True":
        logout = st.sidebar.button("Log Out", type="primary", use_container_width=True)
        if logout:
            st.session_state.cookies["logged_in"] = "False"
            st.session_state.cookies["username"] = "False"
            st.session_state.cookies.save()
            st.rerun()
        
        return True
    else:
        st.title("🔓 Log In - Thrive AI")

        with st.form("login_form"):
            username = st.text_input("Username", value="ThriveAI")
            password = st.text_input("Password", type="password", value="AIThrive")
            submit_button = st.form_submit_button("Login", type="primary")

            if submit_button:
                # TODO: Go to database and check for actual users
                # TODO: Save the user in the cookie
                if username == "ThriveAI" and password == "AIThrive":
                    st.session_state.cookies["logged_in"] = "True"
                    st.session_state.cookies["username"] = username
                    st.session_state.cookies.save()
                    st.rerun()
                else:
                    st.error("Incorrect username or password. Please try again.")
        st.stop()
        return False