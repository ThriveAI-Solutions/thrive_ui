"""Role-based gate for surfacing executed SQL + raw row data in the UI.

Reads [agent].expose_query_details_to from secrets — a list of role
names (admin / doctor / nurse / patient). Defaults to ["admin"].

Stakeholders may want to add doctors to the list once the UX is proven.
The gate keeps that a config change, not a code change.
"""

from __future__ import annotations

from unittest.mock import patch

from agent.observability_gate import role_can_see_query_details
from orm.models import RoleTypeEnum


def _with_secrets(d):
    return patch("streamlit.secrets", d)


def test_default_admin_only_when_no_config():
    with _with_secrets({}):
        assert role_can_see_query_details(RoleTypeEnum.ADMIN) is True
        assert role_can_see_query_details(RoleTypeEnum.DOCTOR) is False
        assert role_can_see_query_details(RoleTypeEnum.NURSE) is False
        assert role_can_see_query_details(RoleTypeEnum.PATIENT) is False


def test_explicit_list_admin_only():
    with _with_secrets({"agent": {"expose_query_details_to": ["admin"]}}):
        assert role_can_see_query_details(RoleTypeEnum.ADMIN) is True
        assert role_can_see_query_details(RoleTypeEnum.DOCTOR) is False


def test_explicit_list_admin_and_doctor():
    with _with_secrets({"agent": {"expose_query_details_to": ["admin", "doctor"]}}):
        assert role_can_see_query_details(RoleTypeEnum.ADMIN) is True
        assert role_can_see_query_details(RoleTypeEnum.DOCTOR) is True
        assert role_can_see_query_details(RoleTypeEnum.NURSE) is False


def test_role_name_match_is_case_insensitive():
    with _with_secrets({"agent": {"expose_query_details_to": ["DOCTOR", "Admin"]}}):
        assert role_can_see_query_details(RoleTypeEnum.ADMIN) is True
        assert role_can_see_query_details(RoleTypeEnum.DOCTOR) is True


def test_handles_int_role_value():
    """Session state often stores user_role as the enum's int value, not
    the enum itself. The gate must accept both."""
    with _with_secrets({"agent": {"expose_query_details_to": ["admin"]}}):
        assert role_can_see_query_details(RoleTypeEnum.ADMIN.value) is True
        assert role_can_see_query_details(RoleTypeEnum.DOCTOR.value) is False


def test_handles_none_role():
    """If user role can't be determined (e.g. anonymous / pre-login), gate denies."""
    with _with_secrets({"agent": {"expose_query_details_to": ["admin"]}}):
        assert role_can_see_query_details(None) is False
