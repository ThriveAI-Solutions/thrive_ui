"""Okta OIDC SSO support.

This module owns all OIDC-side concerns. It is loaded only when
`auth.mode == "oidc"` in `secrets.toml`; in local mode the existing
`utils/auth.py` flow runs unchanged.

See docs/superpowers/specs/2026-05-01-okta-oidc-integration-design.md
for the full design.

Epic #179 added NOT NULL constraints to ``thrive_user.email``,
``thrive_user.organization``, and ``thrive_user.user_role_id``. The JIT
provisioning path in :func:`sync_okta_user_to_db` now enforces the
"fallback defaults with logging" strategy approved during scoping:

  - ``email``         — hard requirement. If the IdP claim doesn't
    provide one, raise :class:`OidcProvisioningError` (the only
    behavioural regression to SSO login, and only when the IdP is
    misconfigured).
  - ``organization``  — derived from the email domain when the claim is
    missing (``alice@thrive.com`` → ``"thrive"``). Logged at WARN with
    ``okta_sub``, ``email``, and the derived value.
  - ``user_role_id`` — defaults to PATIENT when the groups claim doesn't
    resolve to a known group (existing behaviour was DOCTOR; that
    remains for *bad* groups, but a *missing* role / no-group claim now
    routes through the new fallback for symmetry with the migration's
    backfill).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Mapping
from typing import Any, Iterable

from sqlalchemy.orm import Session as SqlSession

from orm.models import RoleTypeEnum, UserRole

logger = logging.getLogger(__name__)


class OidcProvisioningError(RuntimeError):
    """Raised when JIT provisioning cannot proceed for a structural reason.

    The Epic #179 path raises this when the IdP claims do not include an
    ``email``. There is no sensible fallback for a missing email (it's
    the user's canonical identity), so SSO login fails with an
    actionable "contact admin" message instead of silently provisioning
    a half-built user.
    """


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

# Server-side registry of recent auto-login attempts, keyed by a fingerprint
# of the browser's existing cookies (st.context.cookies). This is the only
# marker that reliably survives the Okta roundtrip:
#   - session state dies on every bounce (each return is a fresh session), so
#     it cannot stop a redirect loop when the IdP errors the callback (Okta
#     access_denied "User is not assigned to the client application" looped
#     ~1/second live on 2026-07-14);
#   - writing our own cookie races the redirect against the cookie-component
#     iframe flush, and on prod latency the login redirect got dropped
#     (2026-07-15: every visitor landed on the fallback page instead of Okta).
# The browser's own cookies (_streamlit_xsrf et al.) ride along on every
# connect with zero client-side writes. Entries are pruned on access; a
# process restart forgets attempts, costing at most one extra roundtrip.
_AUTO_LOGIN_ATTEMPTS: dict[str, float] = {}
_AUTO_LOGIN_ATTEMPTS_LOCK = threading.Lock()

# How long a failed attempt (or a logout) suppresses auto-login. This must
# only stop MACHINE-speed loops: IdP error-bounces arrive ~1/second, so ~10s
# bounds those storms — while a human who logs into the portal and clicks
# back to the app (>10s roundtrip) is never suppressed. At 60s this trapped
# post-logout users in an app↔portal bounce until the window expired.
AUTO_LOGIN_RETRY_WINDOW_S = 10.0

# Session-state flag: this session already started an auto-login attempt.
# Used to re-issue st.login() on reruns of the originating session (the
# redirect message can be dropped by an interrupting rerun) — NOT as the
# loop guard; that's the cookie-fingerprint registry, because session state
# does not survive the roundtrip to the IdP.
AUTO_LOGIN_PENDING_KEY = "_oidc_auto_login_pending"

# How many times one session may issue st.login(). One initial issue plus one
# retry covers the dropped-redirect case. It MUST be bounded: with
# prompt=none the whole roundtrip is redirects, so the originating page never
# unloads and its session keeps rerunning — unlimited re-issue navigated the
# tab away from its own in-flight /oauth2callback token exchange (nginx 499)
# in an endless loop (live 2026-07-15). Past the cap, the navigation is in
# flight; render a passive page and let it land.
AUTO_LOGIN_MAX_ISSUES = 2
AUTO_LOGIN_ISSUE_COUNT_KEY = "_oidc_auto_login_issues"

# Query param set by our own logout redirect. Its presence suppresses
# auto-login — st.logout() clears only our cookie, not Okta's session, so
# without this a logout would silently sign the user straight back in.
LOGGED_OUT_QUERY_PARAM = "logged_out"


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


def _organization_from_email(email: str) -> str:
    """Derive an organization name from an email domain.

    ``alice@thrive.com`` → ``"thrive"``. The part before the first dot
    of the host, lowercased. Falls back to ``"unknown"`` for malformed
    inputs (no ``@``, no dot, empty host) — that's only reachable when
    the email is itself ill-formed, which the validator catches
    elsewhere.
    """
    if "@" not in email:
        return "unknown"
    host = email.split("@", 1)[1].strip().lower()
    if not host:
        return "unknown"
    return host.split(".", 1)[0] or "unknown"


def sync_okta_user_to_db(claims: dict, session: SqlSession):
    """Look up or JIT-create a User row matching the OIDC claims.

    Args:
        claims: OIDC ID-token claims dict. Must include ``sub`` and
            ``email`` (Epic #179 — email is the user's canonical
            identity, no fallback). Should also include ``given_name``,
            ``family_name``, ``groups``, and optionally ``organization``
            / ``org``.
        session: Active SQLAlchemy session.

    Returns:
        The User row, with role refreshed from the group claim.

    Raises:
        ValueError: When ``sub`` is missing from claims.
        OidcProvisioningError: When ``email`` is missing from claims.
            SSO login fails with an actionable message rather than
            silently provisioning a row that would violate the Epic
            #179 NOT NULL constraint on ``email``.
    """
    from sqlalchemy import func

    from orm.models import User

    sub = claims.get("sub")
    if not sub:
        raise ValueError("OIDC claims missing required 'sub' field")
    email = (claims.get("email") or "").strip()
    if not email:
        # Hard error per Epic #179 — email is the canonical identity and
        # there is no defensible default. SSO login should surface this
        # to the user with an actionable "contact admin" message.
        raise OidcProvisioningError("OIDC IdP did not provide an `email` claim — contact your administrator.")
    given_name = claims.get("given_name") or ""
    family_name = claims.get("family_name") or ""
    groups = normalize_groups_claim(claims.get("groups"))

    # Role resolution — existing behaviour kept (role_id_from_groups picks
    # the highest-privilege matching group, or DOCTOR if no match).
    # Per Epic #179 the JIT fallback for *missing* groups is PATIENT;
    # apply it only when the claim is empty / absent, so misconfigured
    # group strings still get the existing DOCTOR default behaviour.
    raw_groups_claim = claims.get("groups")
    if raw_groups_claim in (None, "", [], (), set()):
        target_role = session.query(UserRole).filter(UserRole.role == RoleTypeEnum.PATIENT).one_or_none()
        if target_role is None:
            # No PATIENT row — fall back to whatever role_id_from_groups
            # gives us (which itself will pick a default).
            target_role_id = role_id_from_groups(groups, session)
        else:
            target_role_id = target_role.id
        logger.warning(
            "OIDC JIT fallback: role defaulted to PATIENT (no groups claim). okta_sub=%s email=%s",
            sub,
            email,
        )
    else:
        target_role_id = role_id_from_groups(groups, session)

    # Organization resolution — derive from email domain when the claim
    # doesn't provide one. Log at WARN with okta_sub + email so admins
    # can audit. Accept either ``organization`` or ``org`` as the claim
    # name for forwards compatibility with HeL's eventual mapping.
    organization = (claims.get("organization") or claims.get("org") or "").strip()
    if not organization:
        organization = _organization_from_email(email)
        logger.warning(
            "OIDC JIT fallback: organization derived from email domain. okta_sub=%s email=%s fallback_organization=%s",
            sub,
            email,
            organization,
        )

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
            email=email,
            organization=organization,
            okta_sub=sub,
            first_name=given_name,
            last_name=family_name,
            user_role_id=target_role_id,
        )
        session.add(user)
        logger.info("JIT-created OIDC user sub=%s email=%s organization=%s", sub, email, organization)
    else:
        # Existing user: refresh attributes from claims. Per spec §6,
        # Okta is source of truth for OIDC users — role gets overwritten.
        # Organization is *not* refreshed from the fallback (an admin
        # may have set a real value); only refreshed if the claim
        # actually carried a non-empty organization.
        if email and user.email != email:
            user.email = email
        if given_name and user.first_name != given_name:
            user.first_name = given_name
        if family_name and user.last_name != family_name:
            user.last_name = family_name
        if not user.organization:
            # Existing row has no organization (e.g. pre-#179 legacy
            # row not yet backfilled in this session). Fill it now.
            user.organization = organization
        elif claims.get("organization") or claims.get("org"):
            # The IdP supplied a real value — accept it as source of truth.
            user.organization = organization
        user.user_role_id = target_role_id

    from sqlalchemy.exc import IntegrityError

    try:
        session.commit()
    except IntegrityError:
        # Concurrent first-login winner stole the okta_sub or email — roll
        # back and re-query for the now-existing row instead of crashing.
        session.rollback()
        user = session.query(User).filter(User.okta_sub == sub).one_or_none()
        if user is None and email:
            user = session.query(User).filter(func.lower(User.email) == email.lower()).one_or_none()
        if user is None:
            raise
        logger.info("Recovered from JIT race for sub=%s — using existing row id=%s", sub, user.id)
        # Apply the same per-login refresh the non-race path applies.
        if email and user.email != email:
            user.email = email
        if given_name and user.first_name != given_name:
            user.first_name = given_name
        if family_name and user.last_name != family_name:
            user.last_name = family_name
        if not user.okta_sub:
            user.okta_sub = sub
        if not user.organization:
            # Same backfill rule as the non-race path: if the winner
            # left organization blank, fill it from the derived value.
            user.organization = organization
        elif claims.get("organization") or claims.get("org"):
            user.organization = organization
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


def _stable_xsrf_token(raw: str) -> str | None:
    """Recover the stable token from a Tornado XSRF cookie value.

    Tornado v2 tokens ("2|<mask>|<masked_token>|<timestamp>") are re-masked
    on EVERY response — the raw cookie value changes across the OIDC
    roundtrip even though the underlying token is constant (verified live
    2026-07-15). XOR-ing the mask back out recovers the stable token; no
    secret is involved, the mask is in the cookie itself. Plain (v1) tokens
    contain no "|" and are already stable. Returns None for unrecognized
    formats — treating those as stable would silently unbound the retry loop.
    """
    if "|" not in raw:
        return raw
    parts = raw.split("|")
    if len(parts) != 4 or parts[0] != "2":
        return None
    try:
        mask = bytes.fromhex(parts[1])
        masked = bytes.fromhex(parts[2])
        if not mask:
            return None
        return bytes(b ^ mask[i % len(mask)] for i, b in enumerate(masked)).hex()
    except ValueError:
        return None


# Cookies whose values change across the OIDC roundtrip (the OAuth state
# cookie rotates per attempt; auth/user cookies appear on login). They must
# never feed the fingerprint fallback.
_VOLATILE_COOKIE_PREFIXES = ("_streamlit_",)


def _browser_fingerprint() -> str | None:
    """Stable per-browser key derived from the cookies the browser already has.

    Prefers the unmasked Streamlit XSRF token (random per browser, constant
    across the OIDC roundtrip); falls back to a hash of the non-volatile
    cookies. Returns None when no stable identity exists — such a browser's
    retry loop could not be bounded, so callers must not auto-login.
    """
    import hashlib

    import streamlit as st

    try:
        cookies = dict(st.context.cookies)
    except Exception:  # pragma: no cover - context unavailable (bare mode)
        return None
    raw: str | None = None
    xsrf = cookies.get("_streamlit_xsrf") or cookies.get("_xsrf")
    if xsrf:
        raw = _stable_xsrf_token(xsrf)
    if raw is None:
        stable = {k: v for k, v in cookies.items() if not any(k.startswith(p) for p in _VOLATILE_COOKIE_PREFIXES)}
        if not stable:
            return None
        raw = "|".join(f"{k}={v}" for k, v in sorted(stable.items()))
    return hashlib.sha256(raw.encode()).hexdigest()


def _mark_auto_login_attempt() -> None:
    """Record an auto-login attempt for this browser (prunes stale entries)."""
    import time

    fingerprint = _browser_fingerprint()
    if fingerprint is None:
        return
    now = time.time()
    with _AUTO_LOGIN_ATTEMPTS_LOCK:
        for key, ts in list(_AUTO_LOGIN_ATTEMPTS.items()):
            if now - ts >= AUTO_LOGIN_RETRY_WINDOW_S:
                del _AUTO_LOGIN_ATTEMPTS[key]
        _AUTO_LOGIN_ATTEMPTS[fingerprint] = now


def _recent_auto_login_attempt() -> bool:
    """True when this browser attempted auto-login within the retry window."""
    import time

    fingerprint = _browser_fingerprint()
    if fingerprint is None:
        return False
    with _AUTO_LOGIN_ATTEMPTS_LOCK:
        ts = _AUTO_LOGIN_ATTEMPTS.get(fingerprint)
    return ts is not None and time.time() - ts < AUTO_LOGIN_RETRY_WINDOW_S


def _should_auto_login() -> bool:
    """True when an unauthenticated visit should be sent straight to Okta.

    Auto-login is on by default (``[auth].auto_login = false`` disables it)
    and stands down when the user just logged out (``?logged_out=1``), when
    this browser attempted auto-login within the retry window (the IdP just
    bounced us back unauthenticated — retrying would loop), or when the
    browser is fingerprint-less (no cookies — a loop could not be bounded).
    """
    import streamlit as st

    auth = auth_secrets_section()
    if auth is not None and not auth.get("auto_login", True):
        return False
    try:
        if st.query_params.get(LOGGED_OUT_QUERY_PARAM):
            return False
    except Exception:  # pragma: no cover - query params unavailable (bare mode)
        pass
    if _browser_fingerprint() is None:
        return False
    return not _recent_auto_login_attempt()


def _post_logout_redirect_target() -> str:
    """Where the logout meta-refresh should send the browser.

    Redirects that land back on this app (or an unconfigured URL, which
    defaults to the app itself) carry ``?logged_out=1`` so auto-login does
    not immediately sign the user back in. Foreign URLs (e.g. the HeC
    Portal) are passed through untouched.
    """
    from urllib.parse import urlsplit

    auth = auth_secrets_section()
    configured = (auth.get("post_logout_redirect_url") or "").strip() if auth is not None else ""
    if not configured:
        return f"?{LOGGED_OUT_QUERY_PARAM}=1"
    redirect_uri = (auth.get("redirect_uri") or "").strip() if auth is not None else ""
    if redirect_uri and urlsplit(configured).netloc == urlsplit(redirect_uri).netloc:
        sep = "&" if "?" in configured else "?"
        return f"{configured}{sep}{LOGGED_OUT_QUERY_PARAM}=1"
    return configured


def handle_oidc_auth() -> None:
    """OIDC entry point. Called from utils/auth.check_authenticate when in OIDC mode.

    If the user is not logged in, redirect to Okta immediately (auto-login);
    the SSO button renders only as a fallback — after a logout, after a failed
    auto attempt, or when ``[auth].auto_login = false``.
    If the user is logged in, sync the User row, populate session state, and
    render the sidebar welcome banner + Log Out button (replacing what
    _handle_local_auth does in the local path).
    """
    import streamlit as st

    if not getattr(st.user, "is_logged_in", False):
        # Re-issue st.login() on every rerun of the session that started the
        # attempt: Streamlit drops the auth-redirect message when a queued
        # rerun (e.g. a late cookie-component value) interrupts the run that
        # sent it. A drop implies another rerun is queued, so re-issuing on
        # each run guarantees the last one lands. The bounce-back from the
        # IdP is a NEW session without the pending flag, so the server-side
        # registry still bounds the retry loop.
        if st.session_state.get(AUTO_LOGIN_PENDING_KEY) or _should_auto_login():
            st.session_state[AUTO_LOGIN_PENDING_KEY] = True
            issued = st.session_state.get(AUTO_LOGIN_ISSUE_COUNT_KEY, 0)
            if issued < AUTO_LOGIN_MAX_ISSUES:
                st.session_state[AUTO_LOGIN_ISSUE_COUNT_KEY] = issued + 1
                _mark_auto_login_attempt()
                st.login()
            else:
                # Redirect already issued; the browser is mid-roundtrip.
                # Anything that navigates here (another st.login(), a
                # meta-refresh) would cancel the in-flight token exchange.
                st.info("Signing you in…")
            st.stop()
            return  # for tests where st.stop is mocked
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
        # When the operator points unauthenticated visitors at an external
        # sign-in (the HeC Portal login), redirect instead of rendering the
        # manual button page. Combined with `prompt = "none"` in
        # [auth].client_kwargs, visitors with an active IdP session pass
        # through silently and everyone else lands on the Portal.
        # auto_login = false is an explicit button-first choice and wins.
        auth = auth_secrets_section()
        sso_fallback_url = (auth.get("sso_fallback_url") or "").strip() if auth is not None else ""
        auto_login_enabled = auth is None or auth.get("auto_login", True)
        if sso_fallback_url and auto_login_enabled:
            logger.warning("OIDC sign-in fallback: redirecting unauthenticated visitor to %s", sso_fallback_url)
            st.markdown(
                f'<meta http-equiv="refresh" content="2; url={sso_fallback_url}">',
                unsafe_allow_html=True,
            )
            st.info("Taking you to the HEALTHeCOMMUNITY sign-in…")
            st.markdown(f"[Continue to sign-in]({sso_fallback_url})")
            st.stop()
            return  # for tests where st.stop is mocked

        st.title("🔓 Sign in to HEALTHeINTELLIGENCE")
        try:
            just_logged_out = bool(st.query_params.get(LOGGED_OUT_QUERY_PARAM))
        except Exception:  # pragma: no cover - query params unavailable (bare mode)
            just_logged_out = False
        if _recent_auto_login_attempt() and not just_logged_out:
            # We just tried automatically and came back unauthenticated —
            # the IdP rejected or aborted the login (e.g. user not assigned
            # to the application). Tell the human instead of looping.
            st.warning(
                "Automatic sign-in didn't complete. Click the button to try again. "
                "If this keeps happening, your account may not have access to this "
                "application yet — contact your administrator."
            )
            logger.warning("OIDC auto-login roundtrip returned unauthenticated; showing manual sign-in button")
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

    # 4. Suppress auto-login for this browser for the retry window.
    # st.logout() (RP-initiated Okta logout) lands back on the app root with
    # no query param; without this mark, auto-login would fire immediately
    # and throw the user at an Okta form instead of the configured
    # post-logout destination.
    _mark_auto_login_attempt()

    # 5. Emit a meta-refresh redirect to the post-logout URL. Self-targeted
    # (or unconfigured) URLs carry ?logged_out=1 so auto-login stands down.
    st.markdown(
        f'<meta http-equiv="refresh" content="0; url={_post_logout_redirect_target()}">',
        unsafe_allow_html=True,
    )

    # 6. Drop Streamlit's auth cookie (also triggers RP-initiated IdP logout
    # when the auth server advertises end_session_endpoint).
    st.logout()
