"""Okta OIDC SSO support.

This module owns all OIDC-side concerns. It is loaded only when
`auth.mode == "oidc"` in `secrets.toml`; in local mode the existing
`utils/auth.py` flow runs unchanged.

See docs/superpowers/specs/2026-05-01-okta-oidc-integration-design.md
for the full design.
"""

from __future__ import annotations

import logging
from typing import Iterable

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
        logger.error(
            "UserRole row for %s not found; falling back to first available role", chosen
        )
        role = session.query(UserRole).first()
        if role is None:
            raise RuntimeError("No UserRole rows seeded — DB is uninitialized")
    return role.id


def is_oidc_mode() -> bool:
    """True iff secrets.toml has [auth].mode == 'oidc'.

    Any other value, or a missing [auth] section, means local mode.
    """
    import streamlit as st

    auth_section = st.secrets.get("auth", {}) if hasattr(st, "secrets") else {}
    return auth_section.get("mode") == "oidc"


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
    groups = claims.get("groups") or []

    target_role_id = role_id_from_groups(groups, session)

    # 1. Match by okta_sub (canonical).
    user = session.query(User).filter(User.okta_sub == sub).one_or_none()

    # 2. Bootstrap match by email (case-insensitive) and stamp sub.
    if user is None and email:
        user = (
            session.query(User)
            .filter(func.lower(User.email) == email.lower())
            .one_or_none()
        )
        if user is not None:
            user.okta_sub = sub

    # 3. JIT-create.
    if user is None:
        user = User(
            username=email or sub,         # admin can rename later
            password=None,                 # OIDC-only user
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
