"""Unit tests for utils.okta_auth and the User-model OIDC columns.

Tests use an in-memory SQLite engine and only exercise pure helpers and the
sync_okta_user_to_db function against a scratch DB. No real OIDC traffic
flows in these tests; the full flow is validated manually against an Okta
Developer org per docs/superpowers/specs/2026-05-01-okta-oidc-integration-design.md §11.
"""

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
    # Both must be nullable so existing seeded users keep working.
    assert columns["okta_sub"]["nullable"] is True
    assert columns["email"]["nullable"] is True

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
    from utils.okta_auth import sync_okta_user_to_db

    with in_memory_orm_session() as session:
        user = sync_okta_user_to_db(_claims(groups=["unrelated-group"]), session)

        assert user.id is not None
        assert user.okta_sub == "okta-sub-1"
        assert user.email == "alice@example.com"
        assert user.first_name == "Alice"
        assert user.last_name == "Anderson"
        assert user.password is None
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
    from utils.okta_auth import sync_okta_user_to_db

    with in_memory_orm_session() as session:
        admin_role = session.query(UserRole).filter_by(role_name="Admin").one()

        # Manually insert a pre-provisioned row with email set, sub NULL,
        # admin role (i.e. an admin pre-created this user expecting them to log in).
        pre = User(
            username="alice@example.com",
            password=None,
            email="alice@example.com",
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


def test_sync_okta_user_to_db_email_match_is_case_insensitive(in_memory_orm_session):
    """Existing row with email 'Alice@Example.com' matches claim 'alice@example.com'."""
    from orm.models import User, UserRole
    from utils.okta_auth import sync_okta_user_to_db

    with in_memory_orm_session() as session:
        doctor_role = session.query(UserRole).filter_by(role_name="Doctor").one()
        pre = User(
            username="alice@example.com",
            password=None,
            email="Alice@Example.com",
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
    """If st.user.is_logged_in is False, render a login button and stop the page."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch

    from utils.okta_auth import handle_oidc_auth

    fake_user = SimpleNamespace(is_logged_in=False)
    button_mock = MagicMock(return_value=False)
    login_mock = MagicMock()
    stop_mock = MagicMock(side_effect=SystemExit)

    with (
        patch("streamlit.user", fake_user),
        patch("streamlit.button", button_mock),
        patch("streamlit.login", login_mock),
        patch("streamlit.stop", stop_mock),
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

    fake_user = SimpleNamespace(is_logged_in=False)
    # Button returns True meaning the user clicked it.
    button_mock = MagicMock(return_value=True)
    login_mock = MagicMock()

    with (
        patch("streamlit.user", fake_user),
        patch("streamlit.button", button_mock),
        patch("streamlit.login", login_mock),
        patch("streamlit.stop", MagicMock(side_effect=SystemExit)),
        patch("streamlit.title"),
        patch("streamlit.markdown"),
    ):
        try:
            handle_oidc_auth()
        except SystemExit:
            pass

    login_mock.assert_called_once()


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
