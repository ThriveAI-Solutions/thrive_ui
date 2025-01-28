import streamlit as st

def is_logged_in():
    if st.session_state.get("logged_in") is not True: 
        #TODO: check cookies for logged in status
        return False
    else:
        # --- HIDE LOGIN NAVIGATION ---
        st.markdown(
            """
            <style>
            [data-testid="stSidebarNavItems"] li:first-child {
                display: none;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        # --- HIDE LOGIN NAVIGATION ---
        st.sidebar.button("Log Out", on_click=lambda: st.session_state.pop("logged_in"), use_container_width=True) #TODO: remove from cookies
        return True
