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
