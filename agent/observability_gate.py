"""Role gate for the agentic-mode tool-call observability features
(SQL panel + raw rows panel in the chat tool-call card).

Configured via secrets:

    [agent]
    expose_query_details_to = ["admin"]   # default
    # or to roll out to clinicians:
    # expose_query_details_to = ["admin", "doctor"]

Role names map to RoleTypeEnum members case-insensitively
(admin / doctor / nurse / patient).
"""

from __future__ import annotations

from typing import Optional, Union

from orm.models import RoleTypeEnum


_DEFAULT_ALLOW = ("admin",)


def _allowed_role_names() -> set[str]:
    try:
        import streamlit as st

        raw = st.secrets.get("agent", {}).get("expose_query_details_to", _DEFAULT_ALLOW)
    except Exception:
        raw = _DEFAULT_ALLOW
    return {str(name).strip().lower() for name in raw}


def role_can_see_query_details(role: Optional[Union[RoleTypeEnum, int]]) -> bool:
    """Return True iff the given role is in the configured allow-list.

    Accepts a RoleTypeEnum, the enum's int value (how it usually lives
    in session_state), or None (denied).
    """
    if role is None:
        return False
    if isinstance(role, RoleTypeEnum):
        role_name = role.name.lower()
    else:
        try:
            role_name = RoleTypeEnum(int(role)).name.lower()
        except (ValueError, TypeError):
            return False
    return role_name in _allowed_role_names()
