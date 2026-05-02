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
        updated = sync_okta_user_to_db(
            _claims(email="alice.new@example.com"), session
        )

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
        user = sync_okta_user_to_db(
            _claims(groups=["thriveai-doctor"]), session
        )

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
