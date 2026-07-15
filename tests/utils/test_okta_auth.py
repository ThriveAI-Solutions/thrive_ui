"""Unit tests for utils.okta_auth and the User-model OIDC columns.

Tests use an in-memory SQLite engine and only exercise pure helpers and the
sync_okta_user_to_db function against a scratch DB. No real OIDC traffic
flows in these tests; the full flow is validated manually against an Okta
Developer org per docs/superpowers/specs/2026-05-01-okta-oidc-integration-design.md §11.
"""

import re

from sqlalchemy import create_engine, inspect


def test_user_model_has_okta_sub_and_email_columns():
    """Schema check: User table must expose okta_sub and email columns."""
    from orm.models import Base, User

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)

    inspector = inspect(engine)
    columns = {col["name"]: col for col in inspector.get_columns(User.__tablename__)}

    assert "okta_sub" in columns, "User.okta_sub column missing"
    assert "email" in columns, "User.email column missing"
    # okta_sub stays nullable so local-only users still work.
    assert columns["okta_sub"]["nullable"] is True
    # Per Epic #179 email is now NOT NULL — every user has an email.
    assert columns["email"]["nullable"] is False

    # Both must be unique.
    unique_indexes = inspector.get_unique_constraints(User.__tablename__)
    unique_columns = {col for c in unique_indexes for col in c["column_names"]}
    # SQLAlchemy may register unique=True as either a unique constraint or a
    # unique index depending on backend; check both.
    indexes = inspector.get_indexes(User.__tablename__)
    for idx in indexes:
        if idx.get("unique") and len(idx["column_names"]) == 1:
            unique_columns.add(idx["column_names"][0])

    assert "okta_sub" in unique_columns
    assert "email" in unique_columns


def test_in_memory_orm_session_fixture_seeds_user_roles(in_memory_orm_session):
    """Smoke test: fixture should create the four UserRole rows."""
    from orm.models import UserRole

    with in_memory_orm_session() as session:
        names = {r.role_name for r in session.query(UserRole).all()}
        assert names == {"Admin", "Doctor", "Nurse", "Patient"}


def test_role_id_from_groups_admin_wins(in_memory_orm_session):
    """When user is in admin and doctor groups, ADMIN role is selected."""
    from utils.okta_auth import role_id_from_groups

    with in_memory_orm_session() as session:
        role_id = role_id_from_groups(["thriveai-admin", "thriveai-doctor"], session)

    from orm.models import UserRole

    with in_memory_orm_session() as session:
        admin_role = session.query(UserRole).filter_by(role_name="Admin").one()
        # role_id_from_groups must return the Admin role id from this DB.
        # Note: across two separate `with` blocks above, IDs are stable for the
        # same fixture instance — both yield the same engine.
        assert role_id == admin_role.id


def test_role_id_from_groups_no_match_defaults_to_doctor(in_memory_orm_session):
    """No matching group → default DOCTOR."""
    from orm.models import UserRole
    from utils.okta_auth import role_id_from_groups

    with in_memory_orm_session() as session:
        role_id = role_id_from_groups(["random-group", "another-group"], session)
        doctor_role = session.query(UserRole).filter_by(role_name="Doctor").one()

    assert role_id == doctor_role.id


def test_role_id_from_groups_empty_list_defaults_to_doctor(in_memory_orm_session):
    """Empty groups claim → default DOCTOR."""
    from orm.models import UserRole
    from utils.okta_auth import role_id_from_groups

    with in_memory_orm_session() as session:
        role_id = role_id_from_groups([], session)
        doctor_role = session.query(UserRole).filter_by(role_name="Doctor").one()

    assert role_id == doctor_role.id


def test_role_id_from_groups_nurse_alone(in_memory_orm_session):
    """thriveai-nurse alone → Nurse role."""
    from orm.models import UserRole
    from utils.okta_auth import role_id_from_groups

    with in_memory_orm_session() as session:
        role_id = role_id_from_groups(["thriveai-nurse"], session)
        nurse_role = session.query(UserRole).filter_by(role_name="Nurse").one()

    assert role_id == nurse_role.id


def test_is_oidc_mode_returns_true_when_auth_mode_is_oidc():
    """When [auth].mode = 'oidc', is_oidc_mode() returns True."""
    from unittest.mock import patch

    from utils.okta_auth import is_oidc_mode

    with patch("streamlit.secrets", new={"auth": {"mode": "oidc"}}):
        assert is_oidc_mode() is True


def test_is_oidc_mode_returns_false_when_auth_section_absent():
    """No [auth] section → local mode."""
    from unittest.mock import patch

    from utils.okta_auth import is_oidc_mode

    with patch("streamlit.secrets", new={}):
        assert is_oidc_mode() is False


def test_is_oidc_mode_returns_false_when_mode_is_local():
    """[auth].mode = 'local' → local mode (explicit fallback)."""
    from unittest.mock import patch

    from utils.okta_auth import is_oidc_mode

    with patch("streamlit.secrets", new={"auth": {"mode": "local"}}):
        assert is_oidc_mode() is False


def test_is_oidc_mode_returns_false_when_auth_section_is_not_a_dict():
    """Misconfigured auth = 'string' (not a TOML table) → local mode, no crash."""
    from unittest.mock import patch

    from utils.okta_auth import is_oidc_mode

    with patch("streamlit.secrets", new={"auth": "oidc"}):
        # Misconfiguration: auth is a string instead of a section. Should not crash.
        assert is_oidc_mode() is False


def test_is_oidc_mode_accepts_mapping_like_secrets_sub_section():
    """Streamlit's real secrets sub-section is a Mapping, not a dict.

    Regression: an earlier defensive `isinstance(..., dict)` guard returned
    False for Streamlit's actual Secrets/AttrDict object, which broke OIDC
    mode at runtime even though `[auth].mode = "oidc"` was correctly set.
    """
    from collections.abc import Mapping
    from unittest.mock import patch

    from utils.okta_auth import is_oidc_mode

    class FakeStreamlitSecretsSubsection(Mapping):
        def __init__(self, data):
            self._data = data

        def __getitem__(self, key):
            return self._data[key]

        def __iter__(self):
            return iter(self._data)

        def __len__(self):
            return len(self._data)

    fake_top_level = {"auth": FakeStreamlitSecretsSubsection({"mode": "oidc"})}

    with patch("streamlit.secrets", new=fake_top_level):
        assert is_oidc_mode() is True


def test_auth_secrets_section_returns_mapping_when_valid():
    from unittest.mock import patch

    from utils.okta_auth import auth_secrets_section

    with patch("streamlit.secrets", new={"auth": {"mode": "oidc", "post_logout_redirect_url": "https://x/"}}):
        auth = auth_secrets_section()
        assert auth is not None and auth.get("mode") == "oidc"


def test_auth_secrets_section_none_when_auth_is_scalar():
    from unittest.mock import patch

    from utils.okta_auth import auth_secrets_section

    with patch("streamlit.secrets", new={"auth": "oidc"}):
        assert auth_secrets_section() is None


def test_normalize_groups_claim_list():
    from utils.okta_auth import normalize_groups_claim

    assert normalize_groups_claim([" thriveai-admin ", "", "unknown"]) == ["thriveai-admin", "unknown"]


def test_normalize_groups_claim_tuple():
    from utils.okta_auth import normalize_groups_claim

    assert normalize_groups_claim(("thriveai-nurse",)) == ["thriveai-nurse"]


def test_normalize_groups_claim_comma_separated_string():
    from utils.okta_auth import normalize_groups_claim

    assert normalize_groups_claim("thriveai-doctor , thriveai-admin") == [
        "thriveai-doctor",
        "thriveai-admin",
    ]


def test_normalize_groups_claim_single_string_whitespace_trimmed():
    from utils.okta_auth import normalize_groups_claim

    assert normalize_groups_claim("  thriveai-patient  ") == ["thriveai-patient"]
    assert normalize_groups_claim("   ") == []


def test_normalize_groups_claim_non_sequence_returns_empty():
    from unittest.mock import patch

    import utils.okta_auth as mod

    with patch.object(mod.logger, "warning"):
        from utils.okta_auth import normalize_groups_claim

        assert normalize_groups_claim(12345) == []


def test_sync_okta_user_to_db_accept_groups_as_comma_separated_string(in_memory_orm_session):
    from orm.models import RoleTypeEnum
    from utils.okta_auth import sync_okta_user_to_db

    with in_memory_orm_session() as session:
        user = sync_okta_user_to_db(_claims(groups="random, thriveai-admin"), session)

    assert user.role.role == RoleTypeEnum.ADMIN


def _claims(sub="okta-sub-1", email="alice@example.com", groups=None, **extra):
    """Build a fake OIDC claims dict (the shape st.user.to_dict() returns)."""
    base = {
        "sub": sub,
        "email": email,
        "email_verified": True,
        "given_name": "Alice",
        "family_name": "Anderson",
        "groups": groups if groups is not None else ["thriveai-doctor"],
    }
    base.update(extra)
    return base


def test_sync_okta_user_to_db_jit_creates_new_user(in_memory_orm_session):
    """First-time login: row is JIT-created with default DOCTOR role."""
    from orm.models import RoleTypeEnum, User
    from utils.okta_auth import OIDC_PASSWORD_SENTINEL, sync_okta_user_to_db

    with in_memory_orm_session() as session:
        user = sync_okta_user_to_db(_claims(groups=["unrelated-group"]), session)

        assert user.id is not None
        assert user.okta_sub == "okta-sub-1"
        assert user.email == "alice@example.com"
        assert user.first_name == "Alice"
        assert user.last_name == "Anderson"
        assert user.password == OIDC_PASSWORD_SENTINEL
        assert not re.fullmatch(r"[0-9a-f]{64}", user.password)
        assert user.role.role == RoleTypeEnum.DOCTOR  # default fallback

        # Exactly one User row created.
        assert session.query(User).count() == 1


def test_sync_okta_user_to_db_matches_existing_by_sub(in_memory_orm_session):
    """Second login by the same sub reuses the existing row."""
    from orm.models import User
    from utils.okta_auth import sync_okta_user_to_db

    with in_memory_orm_session() as session:
        first = sync_okta_user_to_db(_claims(), session)
        first_id = first.id

        # Same sub, but the email has changed at the IdP. We still match
        # by sub and accept the new email on the row.
        updated = sync_okta_user_to_db(_claims(email="alice.new@example.com"), session)

        assert updated.id == first_id
        assert updated.email == "alice.new@example.com"
        assert session.query(User).count() == 1


def test_sync_okta_user_to_db_bootstrap_match_by_email_stamps_sub(in_memory_orm_session):
    """Pre-provisioned row (email set, sub NULL) gets sub stamped on first login."""
    from orm.models import User, UserRole
    from utils.okta_auth import OIDC_PASSWORD_SENTINEL, sync_okta_user_to_db

    with in_memory_orm_session() as session:
        admin_role = session.query(UserRole).filter_by(role_name="Admin").one()

        # Manually insert a pre-provisioned row with email set, sub NULL,
        # admin role (i.e. an admin pre-created this user expecting them to log in).
        # Epic #179: organization is NOT NULL — admin pre-set "OrgAcme" here.
        pre = User(
            username="alice@example.com",
            password=OIDC_PASSWORD_SENTINEL,
            email="alice@example.com",
            organization="OrgAcme",
            okta_sub=None,
            first_name="Alice",
            last_name="Anderson",
            user_role_id=admin_role.id,
        )
        session.add(pre)
        session.commit()
        pre_id = pre.id

        # Now Alice logs in. Her group claim says doctor, but admin pre-set
        # her role. Per spec §6, Okta is source of truth for OIDC users —
        # her role gets refreshed from the claim on every login.
        user = sync_okta_user_to_db(_claims(groups=["thriveai-doctor"]), session)

        assert user.id == pre_id
        assert user.okta_sub == "okta-sub-1"  # sub now stamped onto pre row
        assert session.query(User).count() == 1


def test_sync_okta_user_to_db_role_updates_on_subsequent_login(in_memory_orm_session):
    """If groups change between logins, the role updates."""
    from orm.models import RoleTypeEnum
    from utils.okta_auth import sync_okta_user_to_db

    with in_memory_orm_session() as session:
        # First login as doctor.
        user = sync_okta_user_to_db(_claims(groups=["thriveai-doctor"]), session)
        assert user.role.role == RoleTypeEnum.DOCTOR

        # User is later promoted to admin in Okta. Next login sees the new group.
        user = sync_okta_user_to_db(_claims(groups=["thriveai-admin"]), session)
        assert user.role.role == RoleTypeEnum.ADMIN


def test_sync_okta_user_to_db_handles_jit_race_via_integrity_error(in_memory_orm_session):
    """Concurrent first-login race: loser rolls back and re-queries instead of crashing.

    Two reruns / browser tabs / processes can pass the okta_sub lookup at the same
    time; one wins the unique-constraint race and the other must recover gracefully
    by re-querying for the now-existing row.
    """
    from sqlalchemy.exc import IntegrityError

    from orm.models import User, UserRole
    from utils.okta_auth import OIDC_PASSWORD_SENTINEL, sync_okta_user_to_db

    SessionLocal = in_memory_orm_session

    with SessionLocal() as losing_session:
        original_commit = losing_session.commit
        commit_calls = {"n": 0}

        def racing_commit():
            commit_calls["n"] += 1
            if commit_calls["n"] == 1:
                # Simulate a concurrent winner: insert the row out-of-band
                # via a separate session, then raise the IntegrityError that
                # the real DB would raise on our commit.
                with SessionLocal() as winning_session:
                    role = winning_session.query(UserRole).filter_by(role_name="Doctor").one()
                    winning_session.add(
                        User(
                            username="alice-other",
                            password=OIDC_PASSWORD_SENTINEL,
                            email="alice@example.com",
                            organization="ThriveAI",  # required per Epic #179
                            okta_sub="okta-sub-1",
                            first_name="Alice",
                            last_name="Anderson",
                            user_role_id=role.id,
                        )
                    )
                    winning_session.commit()
                raise IntegrityError("UNIQUE constraint failed", None, Exception("conflict"))
            return original_commit()

        losing_session.commit = racing_commit

        user = sync_okta_user_to_db(_claims(), losing_session)

        assert user is not None
        assert user.okta_sub == "okta-sub-1"
        assert user.email == "alice@example.com"

    with SessionLocal() as verify:
        assert verify.query(User).filter(User.okta_sub == "okta-sub-1").count() == 1


def test_sync_okta_user_to_db_email_match_is_case_insensitive(in_memory_orm_session):
    """Existing row with email 'Alice@Example.com' matches claim 'alice@example.com'."""
    from orm.models import User, UserRole
    from utils.okta_auth import OIDC_PASSWORD_SENTINEL, sync_okta_user_to_db

    with in_memory_orm_session() as session:
        doctor_role = session.query(UserRole).filter_by(role_name="Doctor").one()
        pre = User(
            username="alice@example.com",
            password=OIDC_PASSWORD_SENTINEL,
            email="Alice@Example.com",
            organization="ThriveAI",  # required per Epic #179
            okta_sub=None,
            first_name="A",
            last_name="A",
            user_role_id=doctor_role.id,
        )
        session.add(pre)
        session.commit()

        user = sync_okta_user_to_db(_claims(email="alice@example.com"), session)
        assert user.id == pre.id
        assert user.okta_sub == "okta-sub-1"
        assert session.query(User).count() == 1


def test_populate_session_state_from_user_writes_expected_keys(in_memory_orm_session):
    """After population, session state mirrors what local-mode login produces."""
    import json
    from unittest.mock import patch

    from utils.okta_auth import populate_session_state_from_user, sync_okta_user_to_db

    fake_session_state = {}
    fake_cookies = {}

    class FakeCookies:
        def get(self, key):
            return fake_cookies.get(key)

        def __setitem__(self, key, value):
            fake_cookies[key] = value

        def __getitem__(self, key):
            return fake_cookies[key]

        def save(self):
            pass

    fake_session_state["cookies"] = FakeCookies()

    with in_memory_orm_session() as session:
        user = sync_okta_user_to_db(_claims(groups=["thriveai-admin"]), session)

    with patch("streamlit.session_state", fake_session_state):
        populate_session_state_from_user(user)

    assert fake_cookies["user_id"] == json.dumps(user.id)
    assert fake_cookies["role_name"] == "Admin"
    assert fake_session_state["user_role"] == 0  # ADMIN
    assert fake_session_state["username"] == "Alice Anderson"


def test_handle_oidc_auth_shows_login_button_when_not_logged_in(in_memory_orm_session):
    """After a failed auto-login roundtrip (fresh marker), render the button."""
    import time
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch

    from utils.okta_auth import handle_oidc_auth

    fake_user = SimpleNamespace(is_logged_in=False)
    button_mock = MagicMock(return_value=False)
    login_mock = MagicMock()
    stop_mock = MagicMock(side_effect=SystemExit)

    with (
        patch("streamlit.user", fake_user),
        patch("streamlit.session_state", {"cookies": {"oidc_auto_login_at": str(time.time())}}),
        patch("streamlit.query_params", {}),
        patch("streamlit.secrets", new={"auth": {"mode": "oidc"}}),
        patch("streamlit.button", button_mock),
        patch("streamlit.login", login_mock),
        patch("streamlit.stop", stop_mock),
        patch("streamlit.warning", MagicMock()),
        patch("streamlit.title"),
        patch("streamlit.markdown"),
    ):
        try:
            handle_oidc_auth()
        except SystemExit:
            pass

    button_mock.assert_called_once()  # SSO button rendered
    login_mock.assert_not_called()  # not clicked yet
    stop_mock.assert_called_once()


def test_handle_oidc_auth_clicking_button_calls_st_login(in_memory_orm_session):
    """If the user clicks the SSO button, st.login() is called."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch

    from utils.okta_auth import handle_oidc_auth

    import time

    fake_user = SimpleNamespace(is_logged_in=False)
    # Button returns True meaning the user clicked it.
    button_mock = MagicMock(return_value=True)
    login_mock = MagicMock()

    with (
        patch("streamlit.user", fake_user),
        patch("streamlit.session_state", {"cookies": {"oidc_auto_login_at": str(time.time())}}),
        patch("streamlit.query_params", {}),
        patch("streamlit.secrets", new={"auth": {"mode": "oidc"}}),
        patch("streamlit.button", button_mock),
        patch("streamlit.login", login_mock),
        patch("streamlit.stop", MagicMock(side_effect=SystemExit)),
        patch("streamlit.warning", MagicMock()),
        patch("streamlit.title"),
        patch("streamlit.markdown"),
    ):
        try:
            handle_oidc_auth()
        except SystemExit:
            pass

    login_mock.assert_called_once()
    button_mock.assert_called_once()  # login came from the click, not auto


def test_handle_oidc_auth_when_logged_in_runs_sync_and_populates_state(
    in_memory_orm_session,
):
    """If logged in, sync the user and populate session state, and draw sidebar."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch

    fake_user = SimpleNamespace(
        is_logged_in=True,
        sub="okta-sub-99",
        email="bob@example.com",
        given_name="Bob",
        family_name="Brown",
        groups=["thriveai-doctor"],
    )

    # st.user supports .to_dict(); add it as a method.
    fake_user.to_dict = lambda: {
        "sub": "okta-sub-99",
        "email": "bob@example.com",
        "email_verified": True,
        "given_name": "Bob",
        "family_name": "Brown",
        "groups": ["thriveai-doctor"],
    }

    fake_session_state = {"cookies": MagicMock()}
    sidebar_mock = MagicMock()
    # st.sidebar.columns returns a list of column-context-managers.
    cm1, cm2 = MagicMock(), MagicMock()
    cm1.__enter__ = MagicMock(return_value=cm1)
    cm1.__exit__ = MagicMock(return_value=False)
    cm2.__enter__ = MagicMock(return_value=cm2)
    cm2.__exit__ = MagicMock(return_value=False)
    sidebar_mock.columns.return_value = [cm1, cm2]

    button_mock = MagicMock(return_value=False)  # logout not clicked

    with (
        patch("streamlit.user", fake_user),
        patch("streamlit.session_state", fake_session_state),
        patch("streamlit.sidebar", sidebar_mock),
        patch("streamlit.title"),
        patch("streamlit.button", button_mock),
        patch("orm.functions.set_user_preferences_in_session_state", MagicMock()),
    ):
        from utils.okta_auth import handle_oidc_auth

        handle_oidc_auth()

    # cookies["role_name"] was written (sync + populate ran).
    fake_session_state["cookies"].__setitem__.assert_any_call("role_name", "Doctor")
    # Sidebar columns were created (welcome + logout button rendered).
    sidebar_mock.columns.assert_called_once()
    # Logout button was rendered (returned False, so logout did not fire).
    button_mock.assert_called()


def test_handle_oidc_auth_repeated_reruns_log_login_once(in_memory_orm_session):
    """A logged-in OIDC session reruns often, but should emit one login event."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch

    fake_user = SimpleNamespace(is_logged_in=True)
    fake_user.to_dict = lambda: {
        "sub": "okta-sub-99",
        "email": "bob@example.com",
        "email_verified": True,
        "given_name": "Bob",
        "family_name": "Brown",
        "groups": ["thriveai-doctor"],
    }

    class FakeCookies(dict):
        def __init__(self):
            super().__init__()
            self.save = MagicMock()

    fake_session_state = {"cookies": FakeCookies()}
    sidebar_mock = MagicMock()
    cm1, cm2 = MagicMock(), MagicMock()
    for cm in (cm1, cm2):
        cm.__enter__ = MagicMock(return_value=cm)
        cm.__exit__ = MagicMock(return_value=False)
    sidebar_mock.columns.return_value = [cm1, cm2]

    log_login_mock = MagicMock()

    with (
        patch("streamlit.user", fake_user),
        patch("streamlit.session_state", fake_session_state),
        patch("streamlit.sidebar", sidebar_mock),
        patch("streamlit.title"),
        patch("streamlit.button", MagicMock(return_value=False)),
        patch("orm.functions.set_user_preferences_in_session_state", MagicMock()),
        patch("orm.logging_functions.log_login", log_login_mock),
    ):
        from utils.okta_auth import handle_oidc_auth

        handle_oidc_auth()
        handle_oidc_auth()

    log_login_mock.assert_called_once_with(user_id=1, username="bob@example.com", success=True)
    fake_session_state["cookies"].save.assert_called_once()


def test_handle_oidc_auth_logout_button_calls_handle_oidc_logout(in_memory_orm_session):
    """Clicking the sidebar Log Out button calls handle_oidc_logout."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch

    fake_user = SimpleNamespace(is_logged_in=True)
    fake_user.to_dict = lambda: {
        "sub": "okta-sub-99",
        "email": "bob@example.com",
        "email_verified": True,
        "given_name": "Bob",
        "family_name": "Brown",
        "groups": ["thriveai-doctor"],
    }

    fake_session_state = {"cookies": MagicMock()}
    sidebar_mock = MagicMock()
    cm1, cm2 = MagicMock(), MagicMock()
    for cm in (cm1, cm2):
        cm.__enter__ = MagicMock(return_value=cm)
        cm.__exit__ = MagicMock(return_value=False)
    sidebar_mock.columns.return_value = [cm1, cm2]

    button_mock = MagicMock(return_value=True)  # user clicked Log Out
    logout_mock = MagicMock()

    with (
        patch("streamlit.user", fake_user),
        patch("streamlit.session_state", fake_session_state),
        patch("streamlit.sidebar", sidebar_mock),
        patch("streamlit.title"),
        patch("streamlit.button", button_mock),
        patch("utils.okta_auth.handle_oidc_logout", logout_mock),
        patch("orm.functions.set_user_preferences_in_session_state", MagicMock()),
    ):
        from utils.okta_auth import handle_oidc_auth

        handle_oidc_auth()

    logout_mock.assert_called_once()


def test_handle_oidc_logout_clears_state_and_calls_st_logout(in_memory_orm_session):
    """Logout clears VannaService cache and session state, emits redirect, calls st.logout()."""
    from unittest.mock import MagicMock, patch

    fake_session_state = {
        "cookies": MagicMock(),
        "messages": ["msg1"],
        "_vn_instance": MagicMock(),
        "selected_llm_provider": "anthropic",
        "selected_llm_model": "claude-3",
        "user_role": 1,
    }
    fake_session_state["cookies"].get.return_value = "42"

    logout_mock = MagicMock()
    invalidate_mock = MagicMock()
    markdown_mock = MagicMock()

    with (
        patch("streamlit.session_state", fake_session_state),
        patch("streamlit.logout", logout_mock),
        patch("streamlit.markdown", markdown_mock),
        patch("streamlit.secrets", new={"auth": {"post_logout_redirect_url": "https://portal.example/"}}),
        patch("utils.vanna_calls.VannaService.invalidate_cache_for_user", invalidate_mock),
    ):
        from utils.okta_auth import handle_oidc_logout

        handle_oidc_logout()

    invalidate_mock.assert_called_once_with("42", 1)
    logout_mock.assert_called_once()
    assert fake_session_state["messages"] == []
    assert fake_session_state["_vn_instance"] is None
    assert fake_session_state["selected_llm_provider"] is None
    assert fake_session_state["selected_llm_model"] is None
    # Redirect HTML was emitted before st.logout().
    redirect_call_found = any("https://portal.example/" in str(call) for call in markdown_mock.call_args_list)
    assert redirect_call_found, "expected a redirect markdown to the post_logout_redirect_url"


def test_handle_oidc_logout_tolerates_scalar_auth_section(in_memory_orm_session):
    """Misconfigured secrets ``auth`` as a scalar must not raise on logout (no redirect)."""
    from unittest.mock import MagicMock, patch

    fake_session_state = {
        "cookies": MagicMock(),
        "messages": [],
        "user_role": 1,
    }
    fake_session_state["cookies"].get.return_value = "1"

    with (
        patch("streamlit.session_state", fake_session_state),
        patch("streamlit.logout"),
        patch("streamlit.markdown"),
        patch("streamlit.secrets", new={"auth": "oidc"}),
        patch("utils.vanna_calls.VannaService.invalidate_cache_for_user", MagicMock()),
    ):
        from utils.okta_auth import handle_oidc_logout

        handle_oidc_logout()


# ── Epic #179: JIT fallback for missing organization / role / email ──────


def test_sync_okta_user_to_db_missing_email_raises_oidc_provisioning_error(in_memory_orm_session):
    """Epic #179: email is canonical identity. Missing → hard error, not silent JIT."""
    import pytest

    from orm.models import User
    from utils.okta_auth import OidcProvisioningError, sync_okta_user_to_db

    with in_memory_orm_session() as session:
        with pytest.raises(OidcProvisioningError) as exc:
            sync_okta_user_to_db(_claims(email=""), session)
        # Message is the actionable one — "contact your administrator."
        assert "contact your administrator" in str(exc.value).lower() or "administrator" in str(exc.value).lower()

        # No row got created — JIT halted before insert.
        assert session.query(User).count() == 0


def test_sync_okta_user_to_db_missing_email_field_completely_raises(in_memory_orm_session):
    """Same hard-error path when the claim dict has no ``email`` key at all."""
    import pytest

    from utils.okta_auth import OidcProvisioningError, sync_okta_user_to_db

    claims = {
        "sub": "okta-sub-1",
        "given_name": "Alice",
        "family_name": "Anderson",
        "groups": ["thriveai-doctor"],
        # No "email" key.
    }
    with in_memory_orm_session() as session, pytest.raises(OidcProvisioningError):
        sync_okta_user_to_db(claims, session)


def test_sync_okta_user_to_db_missing_organization_derives_from_email_domain(in_memory_orm_session, caplog):
    """Epic #179 fallback: alice@thrive.com → organization = "thrive", logged WARN."""
    import logging

    from utils.okta_auth import sync_okta_user_to_db

    with in_memory_orm_session() as session, caplog.at_level(logging.WARNING, logger="utils.okta_auth"):
        user = sync_okta_user_to_db(_claims(email="alice@thrive.com"), session)

    assert user.organization == "thrive"
    # WARN log mentions the okta_sub, the email, and the fallback value.
    fallback_logs = [r for r in caplog.records if "organization derived" in r.getMessage()]
    assert fallback_logs, "expected WARN log for the organization fallback"
    log_msg = fallback_logs[0].getMessage()
    assert "okta-sub-1" in log_msg
    assert "alice@thrive.com" in log_msg
    assert "thrive" in log_msg


def test_sync_okta_user_to_db_organization_claim_accepted_when_present(in_memory_orm_session, caplog):
    """When the IdP supplies ``organization``, no fallback fires and no WARN logs."""
    import logging

    from utils.okta_auth import sync_okta_user_to_db

    claims = _claims(email="alice@thrive.com")
    claims["organization"] = "RealCorp"

    with in_memory_orm_session() as session, caplog.at_level(logging.WARNING, logger="utils.okta_auth"):
        user = sync_okta_user_to_db(claims, session)

    assert user.organization == "RealCorp"
    assert not any("organization derived" in r.getMessage() for r in caplog.records)


def test_sync_okta_user_to_db_org_claim_alias_accepted(in_memory_orm_session):
    """``org`` is accepted as an alias for ``organization`` for forwards compat."""
    from utils.okta_auth import sync_okta_user_to_db

    claims = _claims(email="alice@thrive.com")
    claims["org"] = "RealCorp"

    with in_memory_orm_session() as session:
        user = sync_okta_user_to_db(claims, session)

    assert user.organization == "RealCorp"


def test_sync_okta_user_to_db_empty_groups_claim_defaults_to_patient(in_memory_orm_session, caplog):
    """Epic #179 fallback: empty/missing groups → PATIENT role, logged WARN."""
    import logging

    from orm.models import RoleTypeEnum
    from utils.okta_auth import sync_okta_user_to_db

    with in_memory_orm_session() as session, caplog.at_level(logging.WARNING, logger="utils.okta_auth"):
        user = sync_okta_user_to_db(_claims(groups=[]), session)

    assert user.role.role == RoleTypeEnum.PATIENT
    fallback_logs = [r for r in caplog.records if "role defaulted to PATIENT" in r.getMessage()]
    assert fallback_logs, "expected WARN log for the role fallback"
    assert "okta-sub-1" in fallback_logs[0].getMessage()


def test_sync_okta_user_to_db_no_groups_key_defaults_to_patient(in_memory_orm_session, caplog):
    """The claim dict literally has no ``groups`` key → PATIENT fallback."""
    import logging

    from orm.models import RoleTypeEnum
    from utils.okta_auth import sync_okta_user_to_db

    claims = {
        "sub": "okta-sub-1",
        "email": "alice@example.com",
        "given_name": "Alice",
        "family_name": "Anderson",
        # No "groups" key at all.
    }

    with in_memory_orm_session() as session, caplog.at_level(logging.WARNING, logger="utils.okta_auth"):
        user = sync_okta_user_to_db(claims, session)

    assert user.role.role == RoleTypeEnum.PATIENT
    assert any("role defaulted to PATIENT" in r.getMessage() for r in caplog.records)


def test_sync_okta_user_to_db_unmatched_groups_keeps_doctor_default(in_memory_orm_session):
    """Existing behaviour preserved: non-empty but unmatched groups → DOCTOR."""
    from orm.models import RoleTypeEnum
    from utils.okta_auth import sync_okta_user_to_db

    with in_memory_orm_session() as session:
        user = sync_okta_user_to_db(_claims(groups=["unrelated-group"]), session)

    # No regression — existing fallback to DOCTOR for unmatched groups.
    assert user.role.role == RoleTypeEnum.DOCTOR


def test_sync_okta_user_to_db_both_org_and_role_missing_logs_both_fallbacks(in_memory_orm_session, caplog):
    """Both fallbacks fire and both emit their own WARN log."""
    import logging

    from utils.okta_auth import sync_okta_user_to_db

    with in_memory_orm_session() as session, caplog.at_level(logging.WARNING, logger="utils.okta_auth"):
        user = sync_okta_user_to_db(_claims(email="bob@globex.com", groups=[]), session)

    assert user.organization == "globex"  # derived from email domain
    org_logs = [r for r in caplog.records if "organization derived" in r.getMessage()]
    role_logs = [r for r in caplog.records if "role defaulted to PATIENT" in r.getMessage()]
    assert org_logs, "expected the organization fallback log"
    assert role_logs, "expected the role fallback log"


def test_organization_from_email_helper_handles_subdomain():
    """Helper takes the first dotted segment of the host, lowercased."""
    from utils.okta_auth import _organization_from_email

    assert _organization_from_email("alice@thrive.com") == "thrive"
    assert _organization_from_email("alice@Sub.Thrive.Com") == "sub"
    assert _organization_from_email("alice@THRIVE.com") == "thrive"
    # Malformed cases fall back to "unknown" — defensive, not user-facing.
    assert _organization_from_email("no-at-symbol") == "unknown"
    assert _organization_from_email("foo@") == "unknown"


# ── Auto-login (seamless SSO): st.login() fires without a button click ────
#
# The attempt marker lives in a SERVER-SIDE registry keyed by a fingerprint of
# the browser's existing cookies (st.context.cookies). Session state dies on
# every bounce through Okta, and writing our own cookie races the redirect
# against the cookie-component flush (lost on prod 2026-07-15: the login
# redirect was dropped and every visitor landed on the fallback page). The
# browser's own cookies (_streamlit_xsrf et al.) ride along on every connect
# and survive the roundtrip with zero client-side writes.


def _not_logged_in_patches(session_state, query_params, secrets, context_cookies):
    """Common patch set for the not-logged-in handle_oidc_auth paths."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch

    fake_user = SimpleNamespace(is_logged_in=False)
    fake_context = SimpleNamespace(cookies=context_cookies)
    mocks = SimpleNamespace(
        button=MagicMock(return_value=False),
        login=MagicMock(),
        stop=MagicMock(side_effect=SystemExit),
        warning=MagicMock(),
    )
    patches = [
        patch("streamlit.user", fake_user),
        patch("streamlit.context", fake_context),
        patch("streamlit.session_state", session_state),
        patch("streamlit.query_params", query_params),
        patch("streamlit.secrets", new=secrets),
        patch("streamlit.button", mocks.button),
        patch("streamlit.login", mocks.login),
        patch("streamlit.stop", mocks.stop),
        patch("streamlit.warning", mocks.warning),
        patch("streamlit.title"),
        patch("streamlit.markdown"),
    ]
    return patches, mocks


def _run_not_logged_in(query_params, secrets, context_cookies, session_state=None):
    import contextlib

    from utils.okta_auth import handle_oidc_auth

    patches, mocks = _not_logged_in_patches(
        session_state if session_state is not None else {}, query_params, secrets, context_cookies
    )
    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        try:
            handle_oidc_auth()
        except SystemExit:
            pass
    return mocks


def _clear_attempt_registry():
    import utils.okta_auth as okta_auth

    okta_auth._AUTO_LOGIN_ATTEMPTS.clear()


def test_handle_oidc_auth_auto_login_fires_without_button(in_memory_orm_session):
    """First visit from a cookie-bearing browser goes straight to st.login()."""
    _clear_attempt_registry()
    mocks = _run_not_logged_in({}, {"auth": {"mode": "oidc"}}, {"_streamlit_xsrf": "tok-abc"})

    mocks.login.assert_called_once()
    mocks.stop.assert_called_once()
    mocks.button.assert_not_called()


def test_handle_oidc_auth_failed_roundtrip_breaks_loop_and_warns(in_memory_orm_session):
    """The bounce-back lands in a NEW session but the same browser: no retry.

    This is the loop-breaker for IdP callback errors (e.g. Okta access_denied
    "User is not assigned to the client application"). The second call
    simulates the fresh session after the bounce — same context cookies,
    empty session state.
    """
    _clear_attempt_registry()
    cookies = {"_streamlit_xsrf": "tok-abc"}
    _run_not_logged_in({}, {"auth": {"mode": "oidc"}}, cookies)  # attempt marked
    mocks = _run_not_logged_in({}, {"auth": {"mode": "oidc"}}, cookies)  # bounce landing

    mocks.login.assert_not_called()
    mocks.button.assert_called_once()
    mocks.warning.assert_called_once()  # user is told sign-in didn't complete


def test_handle_oidc_auth_stale_attempt_allows_auto_login_again(in_memory_orm_session):
    """Attempts older than the retry window don't block the next visit."""
    import time

    import utils.okta_auth as okta_auth

    _clear_attempt_registry()
    cookies = {"_streamlit_xsrf": "tok-abc"}
    _run_not_logged_in({}, {"auth": {"mode": "oidc"}}, cookies)
    # Age the recorded attempt beyond the retry window.
    for key in list(okta_auth._AUTO_LOGIN_ATTEMPTS):
        okta_auth._AUTO_LOGIN_ATTEMPTS[key] = time.time() - okta_auth.AUTO_LOGIN_RETRY_WINDOW_S - 1

    mocks = _run_not_logged_in({}, {"auth": {"mode": "oidc"}}, cookies)
    mocks.login.assert_called_once()
    mocks.button.assert_not_called()


def test_handle_oidc_auth_different_browsers_do_not_interfere(in_memory_orm_session):
    """A failed attempt in one browser must not suppress another browser."""
    _clear_attempt_registry()
    _run_not_logged_in({}, {"auth": {"mode": "oidc"}}, {"_streamlit_xsrf": "browser-A"})
    mocks = _run_not_logged_in({}, {"auth": {"mode": "oidc"}}, {"_streamlit_xsrf": "browser-B"})

    mocks.login.assert_called_once()
    mocks.button.assert_not_called()


def test_handle_oidc_auth_cookieless_browser_gets_button_not_loop(in_memory_orm_session):
    """No cookies → no way to bound a retry loop → never auto-login."""
    _clear_attempt_registry()
    mocks = _run_not_logged_in({}, {"auth": {"mode": "oidc"}}, {})

    mocks.login.assert_not_called()
    mocks.button.assert_called_once()
    mocks.warning.assert_not_called()  # nothing failed; just not automatable


def test_handle_oidc_auth_no_auto_login_after_logout(in_memory_orm_session):
    """Arriving with ?logged_out=1 (post-logout) must not silently re-login."""
    _clear_attempt_registry()
    mocks = _run_not_logged_in({"logged_out": "1"}, {"auth": {"mode": "oidc"}}, {"_streamlit_xsrf": "tok"})

    mocks.login.assert_not_called()
    mocks.button.assert_called_once()
    mocks.warning.assert_not_called()  # normal logout, nothing went wrong


def test_handle_oidc_auth_auto_login_disabled_by_config(in_memory_orm_session):
    """[auth].auto_login = false restores the button-first behavior."""
    _clear_attempt_registry()
    mocks = _run_not_logged_in({}, {"auth": {"mode": "oidc", "auto_login": False}}, {"_streamlit_xsrf": "tok"})

    mocks.login.assert_not_called()
    mocks.button.assert_called_once()


def test_handle_oidc_logout_appends_logged_out_param_for_self_redirect(in_memory_orm_session):
    """When post-logout lands back on our own host, tag it so auto-login stands down."""
    from unittest.mock import MagicMock, patch

    fake_session_state = {"cookies": MagicMock(), "messages": [], "user_role": 1}
    fake_session_state["cookies"].get.return_value = "1"
    markdown_mock = MagicMock()

    secrets = {
        "auth": {
            "post_logout_redirect_url": "https://wnyhealtheintelligence.com/",
            "redirect_uri": "https://wnyhealtheintelligence.com/oauth2callback",
        }
    }
    with (
        patch("streamlit.session_state", fake_session_state),
        patch("streamlit.logout"),
        patch("streamlit.markdown", markdown_mock),
        patch("streamlit.secrets", new=secrets),
        patch("utils.vanna_calls.VannaService.invalidate_cache_for_user", MagicMock()),
    ):
        from utils.okta_auth import handle_oidc_logout

        handle_oidc_logout()

    redirect_calls = [str(c) for c in markdown_mock.call_args_list if "http-equiv" in str(c)]
    assert redirect_calls, "expected a meta-refresh redirect"
    assert any("logged_out=1" in c for c in redirect_calls)


def test_handle_oidc_logout_foreign_redirect_left_untouched(in_memory_orm_session):
    """A foreign post-logout URL (e.g. the HeC Portal) gets no logged_out param."""
    from unittest.mock import MagicMock, patch

    fake_session_state = {"cookies": MagicMock(), "messages": [], "user_role": 1}
    fake_session_state["cookies"].get.return_value = "1"
    markdown_mock = MagicMock()

    secrets = {
        "auth": {
            "post_logout_redirect_url": "https://portal.example/",
            "redirect_uri": "https://wnyhealtheintelligence.com/oauth2callback",
        }
    }
    with (
        patch("streamlit.session_state", fake_session_state),
        patch("streamlit.logout"),
        patch("streamlit.markdown", markdown_mock),
        patch("streamlit.secrets", new=secrets),
        patch("utils.vanna_calls.VannaService.invalidate_cache_for_user", MagicMock()),
    ):
        from utils.okta_auth import handle_oidc_logout

        handle_oidc_logout()

    redirect_calls = [str(c) for c in markdown_mock.call_args_list if "http-equiv" in str(c)]
    assert redirect_calls, "expected a meta-refresh redirect"
    assert any("https://portal.example/" in c for c in redirect_calls)
    assert not any("logged_out" in c for c in redirect_calls)


def test_handle_oidc_logout_unconfigured_url_still_tags_logged_out(in_memory_orm_session):
    """No post_logout_redirect_url → redirect to the app itself with ?logged_out=1."""
    from unittest.mock import MagicMock, patch

    fake_session_state = {"cookies": MagicMock(), "messages": [], "user_role": 1}
    fake_session_state["cookies"].get.return_value = "1"
    markdown_mock = MagicMock()

    with (
        patch("streamlit.session_state", fake_session_state),
        patch("streamlit.logout"),
        patch("streamlit.markdown", markdown_mock),
        patch("streamlit.secrets", new={"auth": {"mode": "oidc"}}),
        patch("utils.vanna_calls.VannaService.invalidate_cache_for_user", MagicMock()),
    ):
        from utils.okta_auth import handle_oidc_logout

        handle_oidc_logout()

    redirect_calls = [str(c) for c in markdown_mock.call_args_list if "http-equiv" in str(c)]
    assert redirect_calls, "expected a meta-refresh redirect even with no configured URL"
    assert any("logged_out=1" in c for c in redirect_calls)


# ── Fingerprint stability across the OIDC roundtrip ───────────────────────
#
# Real values captured 2026-07-15: Tornado re-masks the v2 XSRF cookie on
# every response (mask + masked token change, underlying token is stable),
# and _streamlit_session (OAuth state) changes per attempt. The fingerprint
# must survive both, or the loop-breaker silently stops working.

_XSRF_SAMPLE_A = "2|ea3264e1|b3e4d8bca17692b73eed0b6074db2d99|1784076427"
_XSRF_SAMPLE_B = "2|a7cdaa61|fe1b163cec895c377312c5e03924e319|1784076427"


def _fingerprint_for(cookies):
    from types import SimpleNamespace
    from unittest.mock import patch

    from utils.okta_auth import _browser_fingerprint

    with patch("streamlit.context", SimpleNamespace(cookies=cookies)):
        return _browser_fingerprint()


def test_browser_fingerprint_stable_across_xsrf_remasking():
    """Differently-masked v2 XSRF cookies must map to the same fingerprint."""
    fp_a = _fingerprint_for({"_streamlit_xsrf": _XSRF_SAMPLE_A})
    fp_b = _fingerprint_for({"_streamlit_xsrf": _XSRF_SAMPLE_B})
    assert fp_a is not None
    assert fp_a == fp_b


def test_browser_fingerprint_differs_for_different_underlying_tokens():
    fp_a = _fingerprint_for({"_streamlit_xsrf": _XSRF_SAMPLE_A})
    fp_other = _fingerprint_for({"_streamlit_xsrf": "2|00000000|deadbeefdeadbeefdeadbeefdeadbeef|1784076427"})
    assert fp_a != fp_other


def test_browser_fingerprint_fallback_ignores_volatile_streamlit_cookies():
    """Without an XSRF cookie, volatile _streamlit_* values must not change the key."""
    fp_1 = _fingerprint_for({"_streamlit_session": "state-attempt-1", "ajs_anonymous_id": "anon-1"})
    fp_2 = _fingerprint_for({"_streamlit_session": "state-attempt-2", "ajs_anonymous_id": "anon-1"})
    assert fp_1 is not None
    assert fp_1 == fp_2


def test_browser_fingerprint_none_when_only_volatile_cookies():
    """Only volatile cookies → no stable identity → None (no auto-login)."""
    assert _fingerprint_for({"_streamlit_session": "state-only"}) is None


def test_handle_oidc_auth_remasked_xsrf_still_breaks_loop(in_memory_orm_session):
    """End-to-end loop-breaker with realistic re-masked XSRF cookies."""
    _clear_attempt_registry()
    _run_not_logged_in({}, {"auth": {"mode": "oidc"}}, {"_streamlit_xsrf": _XSRF_SAMPLE_A})
    mocks = _run_not_logged_in({}, {"auth": {"mode": "oidc"}}, {"_streamlit_xsrf": _XSRF_SAMPLE_B})

    mocks.login.assert_not_called()
    mocks.button.assert_called_once()
    mocks.warning.assert_called_once()


def test_handle_oidc_auth_reissues_login_on_rerun_of_originating_session(in_memory_orm_session):
    """Reruns of the session that started auto-login must re-issue st.login().

    Streamlit drops the auth-redirect message when a queued rerun (e.g. a
    late cookie-component value) interrupts the run that called st.login()
    — observed intermittently local and consistently on prod latency. Each
    rerun re-issuing the login is self-healing: a drop implies another rerun
    is queued, and the last run's redirect always lands. The bounce-back is
    a NEW session (no pending flag), so the loop stays bounded.
    """
    _clear_attempt_registry()
    cookies = {"_streamlit_xsrf": "tok-abc"}
    session_state = {}
    _run_not_logged_in({}, {"auth": {"mode": "oidc"}}, cookies, session_state=session_state)
    assert session_state.get("_oidc_auto_login_pending") is True

    # Rerun of the SAME session (session_state persists): marker is fresh,
    # but the pending flag must win — login again, not the warning page.
    mocks = _run_not_logged_in({}, {"auth": {"mode": "oidc"}}, cookies, session_state=session_state)
    mocks.login.assert_called_once()
    mocks.button.assert_not_called()
    mocks.warning.assert_not_called()
