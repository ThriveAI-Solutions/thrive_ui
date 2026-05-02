"""Okta OIDC SSO support.

This module owns all OIDC-side concerns. It is loaded only when
`auth.mode == "oidc"` in `secrets.toml`; in local mode the existing
`utils/auth.py` flow runs unchanged.

See docs/superpowers/specs/2026-05-01-okta-oidc-integration-design.md
for the full design.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Iterable

from sqlalchemy.orm import Session as SqlSession

from orm.models import RoleTypeEnum, UserRole

logger = logging.getLogger(__name__)


# Group-name → RoleTypeEnum mapping. HeL/dev-Okta must create groups with
# these names and emit them in the `groups` claim of the ID token.
OKTA_GROUP_TO_ROLE: dict[str, RoleTypeEnum] = {
    "thriveai-admin": RoleTypeEnum.ADMIN,
    "thriveai-doctor": RoleTypeEnum.DOCTOR,
    "thriveai-nurse": RoleTypeEnum.NURSE,
    "thriveai-patient": RoleTypeEnum.PATIENT,
}

# When no group in the user's claim matches OKTA_GROUP_TO_ROLE, this role
# is assigned. Per stakeholder guidance ("give everyone more lax permissions
# by default"), the default is DOCTOR.
DEFAULT_ROLE_IF_NO_GROUP_MATCH: RoleTypeEnum = RoleTypeEnum.DOCTOR

# OIDC users authenticate through Okta only. Local auth hashes user input with
# SHA-256 before comparing, so this non-hex sentinel can never match a local
# password login while still satisfying legacy SQLite NOT NULL schemas.
OIDC_PASSWORD_SENTINEL = "__OIDC_AUTH_ONLY__"


def auth_secrets_section() -> Mapping[str, Any] | None:
    """Return Streamlit ``[auth]`` as a mapping, or None if missing or invalid.

    Rejects scalar misconfigurations (e.g. ``auth = "oidc"``) the same way
    ``is_oidc_mode`` does, so callers can safely use ``.get(...)``.
    """
    import streamlit as st

    if not hasattr(st, "secrets"):
        return None
    auth_section = st.secrets.get("auth", {})
    if not hasattr(auth_section, "get"):
        return None
    return auth_section


def normalize_groups_claim(raw: Any) -> list[str]:
    """Coerce the OIDC ``groups`` claim to a list of non-empty group name strings."""
    if raw is None:
        return []
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return []
        if "," in s:
            return [part.strip() for part in s.split(",") if part.strip()]
        return [s]
    if isinstance(raw, (list, tuple, set)):
        out: list[str] = []
        for item in raw:
            if isinstance(item, str):
                stripped = item.strip()
                if stripped:
                    out.append(stripped)
        return out
    logger.warning("Unexpected groups claim type %s — using empty list", type(raw).__name__)
    return []


def role_id_from_groups(groups: Iterable[str], session: SqlSession) -> int:
    """Resolve a list of Okta group names to a UserRole.id in the DB.

    Picks the highest-privilege (lowest RoleTypeEnum value) matching group.
    Falls back to DEFAULT_ROLE_IF_NO_GROUP_MATCH if no group matches.
    """
    matched = [OKTA_GROUP_TO_ROLE[g] for g in groups if g in OKTA_GROUP_TO_ROLE]
    chosen = min(matched, key=lambda r: r.value) if matched else DEFAULT_ROLE_IF_NO_GROUP_MATCH

    role = session.query(UserRole).filter(UserRole.role == chosen).one_or_none()
    if role is None:
        # Defensive: the four UserRole rows should always exist (seeded by
        # orm.models.seed_initial_data). If they're missing, log and fall back.
        logger.error("UserRole row for %s not found; falling back to first available role", chosen)
        role = session.query(UserRole).first()
        if role is None:
            raise RuntimeError("No UserRole rows seeded — DB is uninitialized")
    return role.id


def is_oidc_mode() -> bool:
    """True iff secrets.toml has [auth].mode == 'oidc'.

    Any other value, or a missing [auth] section, means local mode.
    A misconfigured non-dict auth value (e.g. `auth = "oidc"` instead of
    `[auth]\nmode = "oidc"`) is also treated as local mode rather than
    crashing.
    """
    auth = auth_secrets_section()
    return auth is not None and auth.get("mode") == "oidc"


def sync_okta_user_to_db(claims: dict, session: SqlSession):
    """Look up or JIT-create a User row matching the OIDC claims.

    Args:
        claims: OIDC ID-token claims dict. Must include `sub`. Should
            include `email`, `given_name`, `family_name`, and `groups`.
        session: Active SQLAlchemy session.

    Returns:
        The User row, with role refreshed from the group claim.
    """
    from sqlalchemy import func

    from orm.models import User

    sub = claims.get("sub")
    if not sub:
        raise ValueError("OIDC claims missing required 'sub' field")
    email = (claims.get("email") or "").strip()
    given_name = claims.get("given_name") or ""
    family_name = claims.get("family_name") or ""
    groups = normalize_groups_claim(claims.get("groups"))

    target_role_id = role_id_from_groups(groups, session)

    # 1. Match by okta_sub (canonical).
    user = session.query(User).filter(User.okta_sub == sub).one_or_none()

    # 2. Bootstrap match by email (case-insensitive) and stamp sub.
    if user is None and email:
        user = session.query(User).filter(func.lower(User.email) == email.lower()).one_or_none()
        if user is not None:
            user.okta_sub = sub

    # 3. JIT-create.
    if user is None:
        user = User(
            username=email or sub,  # admin can rename later
            password=OIDC_PASSWORD_SENTINEL,
            email=email or None,
            okta_sub=sub,
            first_name=given_name,
            last_name=family_name,
            user_role_id=target_role_id,
        )
        session.add(user)
        logger.info("JIT-created OIDC user sub=%s email=%s", sub, email)
    else:
        # Existing user: refresh attributes from claims. Per spec §6,
        # Okta is source of truth for OIDC users — role gets overwritten.
        if email and user.email != email:
            user.email = email
        if given_name and user.first_name != given_name:
            user.first_name = given_name
        if family_name and user.last_name != family_name:
            user.last_name = family_name
        user.user_role_id = target_role_id

    session.commit()
    session.refresh(user)
    # Force-load the role relationship so the caller can read user.role.
    _ = user.role
    return user


def populate_session_state_from_user(user) -> None:
    """Mirror a User row into session state in the shape downstream code expects.

    After this returns, the app behaves identically to a local-mode login:
    cookies['user_id'], cookies['role_name'], session_state.user_role,
    session_state.username, and all preference flags are populated.
    """
    import json

    import streamlit as st

    from orm.functions import set_user_preferences_in_session_state

    user_id_cookie = json.dumps(user.id)
    role_name_cookie = user.role.role_name
    cookies = st.session_state["cookies"]
    cookies_changed = cookies.get("user_id") != user_id_cookie or cookies.get("role_name") != role_name_cookie
    if cookies_changed:
        cookies["user_id"] = user_id_cookie
        cookies["role_name"] = role_name_cookie
    st.session_state["user_role"] = user.role.role.value
    st.session_state["username"] = f"{user.first_name} {user.last_name}".strip()

    # Mirror local-mode behavior: flush encrypted cookies to the browser so
    # subsequent reruns see the persisted values, not just in-memory state.
    if cookies_changed and hasattr(cookies, "save"):
        try:
            cookies.save()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to save cookies after OIDC populate: %s", exc)

    # Reuse the existing preference loader; it reads cookies['user_id'] and
    # populates the same set of session-state keys local-mode login does.
    try:
        set_user_preferences_in_session_state()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("set_user_preferences_in_session_state failed: %s", exc)


def handle_oidc_auth() -> None:
    """OIDC entry point. Called from utils/auth.check_authenticate when in OIDC mode.

    If the user is not logged in, render a single SSO button and stop the page.
    If the user is logged in, sync the User row, populate session state, and
    render the sidebar welcome banner + Log Out button (replacing what
    _handle_local_auth does in the local path).
    """
    import streamlit as st

    if not getattr(st.user, "is_logged_in", False):
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
        st.title("🔓 Sign in to HEALTHeINTELLIGENCE")
        if st.button("Sign in with HEALTHeCOMMUNITY (Okta)", type="primary"):
            st.login()  # uses [auth] config; for multi-provider use st.login("okta")
        st.stop()
        return  # for tests where st.stop is mocked

    # Logged in. Materialize claims and sync.
    claims = (
        st.user.to_dict()
        if hasattr(st.user, "to_dict")
        else {
            "sub": getattr(st.user, "sub", None),
            "email": getattr(st.user, "email", None),
            "email_verified": getattr(st.user, "email_verified", None),
            "given_name": getattr(st.user, "given_name", ""),
            "family_name": getattr(st.user, "family_name", ""),
            "groups": getattr(st.user, "groups", []),
        }
    )

    from orm.models import SessionLocal

    with SessionLocal() as session:
        user = sync_okta_user_to_db(claims, session)
        populate_session_state_from_user(user)
        # Cache attributes we need for the sidebar before the session closes.
        display_name = f"{user.first_name} {user.last_name}".strip()
        username = user.username
        user_id = user.id

    # Streamlit reruns the script frequently. Only the transition into this
    # authenticated user session is a login event; later reruns are not.
    login_marker_key = "_oidc_logged_login_user_id"
    if st.session_state.get(login_marker_key) != user_id:
        try:
            from orm.logging_functions import log_login

            log_login(user_id=user_id, username=username, success=True)
            st.session_state[login_marker_key] = user_id
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to log OIDC login for user %s: %s", username, exc)

    # Render sidebar welcome banner + Log Out button. This mirrors the local
    # path in utils/auth.py:_handle_local_auth so the user gets the same UI.
    cols = st.sidebar.columns([0.7, 0.3], vertical_alignment="bottom")
    with cols[0]:
        st.title(f"Welcome {display_name}")
    with cols[1]:
        if st.button("Log Out"):
            handle_oidc_logout()


def handle_oidc_logout() -> None:
    """Logout for OIDC mode.

    1. Invalidate VannaService cache for the current user.
    2. Clear app session state to match local-mode logout shape.
    3. Clear the mirrored cookies set in populate_session_state_from_user.
    4. Emit a meta-refresh redirect to auth.post_logout_redirect_url so the
       browser navigates to the Portal as the page tears down.
    5. Call st.logout() to drop Streamlit's auth cookie.

    The redirect is best-effort — if browser timing or Streamlit's rerun
    suppresses the meta-refresh, the user lands on the SSO button page and
    can navigate manually.
    """
    import json

    import streamlit as st

    # 1. Invalidate VannaService cache while we still have the user identity.
    # Use bracket access on session_state so this works against both Streamlit's
    # SessionStateProxy and tests that patch session_state with a plain dict.
    try:
        from utils.vanna_calls import VannaService

        cookies = st.session_state["cookies"] if "cookies" in st.session_state else None
        user_id_str = cookies.get("user_id") if cookies is not None else None
        user_role = st.session_state.get("user_role")
        if user_id_str and user_role is not None:
            # cookies["user_id"] is JSON-encoded for compat with local mode;
            # tolerate both raw and JSON-encoded values.
            try:
                user_id = json.loads(user_id_str)
            except (TypeError, ValueError):
                user_id = user_id_str
            VannaService.invalidate_cache_for_user(str(user_id), user_role)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to invalidate VannaService cache on OIDC logout: %s", exc)

    # 2. Clear app state.
    st.session_state["messages"] = []
    st.session_state["selected_llm_provider"] = None
    st.session_state["selected_llm_model"] = None
    if "_vn_instance" in st.session_state:
        st.session_state["_vn_instance"] = None
    st.session_state.pop("_oidc_logged_login_user_id", None)

    # 3. Clear mirrored cookies.
    try:
        cookies = st.session_state["cookies"] if "cookies" in st.session_state else None
        if cookies is not None:
            cookies["user_id"] = ""
            cookies["role_name"] = ""
            if hasattr(cookies, "save"):
                cookies.save()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to clear mirrored cookies on OIDC logout: %s", exc)

    # 4. Emit a meta-refresh redirect to the post-logout URL.
    auth = auth_secrets_section()
    redirect_url = auth.get("post_logout_redirect_url") if auth is not None else None
    if redirect_url:
        st.markdown(
            f'<meta http-equiv="refresh" content="0; url={redirect_url}">',
            unsafe_allow_html=True,
        )

    # 5. Drop Streamlit's auth cookie.
    st.logout()
