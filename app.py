import streamlit as st

from utils.discord_logging import add_discord_handler_if_configured, initialize_discord_logging_after_streamlit
from utils.logging_config import setup_logging
from utils.quick_logger import INFO, get_logger

# Set the page configuration to wide mode
st.set_page_config(layout="wide")
from streamlit_cookies_manager_ext import EncryptedCookieManager

from utils.auth import check_authenticate

# setup logging
setup_logging(debug=True)

logger = get_logger(__name__)

# Initialize Discord logging after Streamlit is ready

# add_discord_handler_if_configured(logger)

# silence watchdog warnings
get_logger("fsevents").setLevel(INFO)
get_logger("chromadb").setLevel(INFO)
get_logger("httpcore").setLevel(INFO)
get_logger("httpx").setLevel(INFO)


# Run pending DB migrations and seed defaults exactly once per process.
# Streamlit reruns app.py on every interaction, so cache_resource pins this
# to first boot. cache_resource does not cache exceptions, so we trap and
# cache a sentinel ourselves to avoid hammering alembic on every rerun when
# something is broken.
@st.cache_resource
def _bootstrap_db() -> Exception | None:
    from orm.models import init_db

    try:
        init_db()
    except Exception as exc:
        logger.critical("Database initialization failed: %s", exc, exc_info=True)
        return exc
    return None


_bootstrap_err = _bootstrap_db()
if _bootstrap_err is not None:
    st.error(f"Database initialization failed: {_bootstrap_err}")
    st.stop()

# Initialize the cookie manager. The prefix is configurable so that multiple
# deployments behind the same hostname (e.g. prod at "/" and a dev instance at
# "/dev") don't read each other's cookies — a shared prefix points one app at a
# user_id that only exists in the other's database. Defaults to "thrive_ai_".
st.session_state.cookies = EncryptedCookieManager(
    prefix=st.secrets["cookie"].get("prefix", "thrive_ai_"),
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

# Conditionally add Admin pages for admins
pages = [chat_bot_page, user_page]
if st.session_state.get("user_role") == 0:  # RoleTypeEnum.ADMIN.value = 0
    analytics_page = st.Page(
        page="views/admin_analytics.py",
        title="Admin Analytics",
        icon="📈",
    )
    feedback_page = st.Page(
        page="views/admin_feedback.py",
        title="Feedback Dashboard",
        icon="💬",
    )
    pages.append(analytics_page)
    pages.append(feedback_page)

pg = st.navigation(pages=pages)

check_authenticate()

initialize_discord_logging_after_streamlit(st.session_state.get("username"))

pg.run()
