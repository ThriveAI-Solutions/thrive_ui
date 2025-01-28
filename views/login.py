import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from pathlib import Path
print('login')
#--- HIDE SIDEBAR ---
# st.markdown(
#     """
# <style>
#     [data-testid="stSidebarCollapsedControl"], [data-testid="stSidebar"] {
#         display: none
#     }
# </style>
# """,
#     unsafe_allow_html=True,
# )
#--- HIDE SIDEBAR ---

# --- AUTHENTICATION SETUP ---
# Construct the path to the config.yaml file relative to the script's location
config_path = Path(__file__).resolve().parent.parent / "config.yaml"

# Load the YAML configuration
with config_path.open("r") as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# Store the authenticator in the session state
if "authenticator" not in st.session_state:
    st.session_state["authenticator"] = authenticator

authenticator.login(location='main')

if st.session_state["authentication_status"]:
    print('goto chat_bot')
    st.write(f'Welcome *{st.session_state["name"]}*')
    st.switch_page("views/chat_bot.py")

elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')

# We call below code in case of registration, reset password, etc.
with open(config_path, 'w') as file:
    yaml.dump(config, file, default_flow_style=False)

# --- AUTHENTICATION SETUP ---