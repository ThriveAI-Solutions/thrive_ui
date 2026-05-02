# Okta OIDC SSO Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate Okta OIDC SSO into the Streamlit `thrive_ui` app, with a config switch that keeps the existing username/password login as a dev/break-glass fallback. Identity flows from Okta into the existing SQLite `User` table via JIT provisioning, with the user's role derived from an Okta group claim (default `DOCTOR`).

**Architecture:** A new `utils/okta_auth.py` module owns OIDC concerns and uses Streamlit's native `st.login()` / `st.user` (Authlib under the hood). `utils/auth.py` becomes a thin dispatcher that branches on `auth.mode` from `secrets.toml`. The existing local-login path is left intact and still runs when `[auth]` is absent. Both paths leave the app in the same session-state shape so downstream code (`VannaService`, RAG filters, page-level admin gates, `extract_user_context_from_streamlit`) works unchanged.

**Tech Stack:** Python 3.13, Streamlit 1.43.2 (native `st.login()` / `st.user`), Authlib (transitive via Streamlit), SQLAlchemy + SQLite, pytest.

**Spec:** `docs/superpowers/specs/2026-05-01-okta-oidc-integration-design.md` (commit `6875e1d`).

---

## File Structure

| File | Disposition | Responsibility |
|---|---|---|
| `orm/models.py` | Modify | Add `okta_sub` (VARCHAR(255), unique nullable) and `email` (VARCHAR(320), unique nullable, `COLLATE NOCASE`) columns to the `User` model. |
| `scripts/migrate_add_okta_columns.py` | Create | Idempotent ALTER TABLE script for existing dev SQLite DBs that pre-date the column additions. |
| `utils/okta_auth.py` | Create | All OIDC-side concerns: `is_oidc_mode`, `role_id_from_groups`, `sync_okta_user_to_db`, `handle_oidc_auth`, `handle_oidc_logout`. Pure helpers are unit-testable; thin Streamlit-bound entry points are tested via mocked `st.user`. |
| `utils/auth.py` | Modify | Dispatcher: `check_authenticate()` checks `is_oidc_mode()` and routes to either the new OIDC handler or the existing local-cookie path (which moves into a private helper). |
| `tests/conftest.py` | Modify | Add a fixture `in_memory_orm_session` that wires SQLAlchemy to `sqlite:///:memory:`, runs `Base.metadata.create_all`, seeds `UserRole`s, and patches `orm.models.SessionLocal` so production code under test uses the in-memory engine. |
| `tests/utils/test_okta_auth.py` | Create | Unit tests for `is_oidc_mode`, `role_id_from_groups`, `sync_okta_user_to_db`. |
| `tests/utils/test_auth_dispatcher.py` | Create | Verifies the dispatcher routes to the local path when `[auth]` is absent and to the OIDC path when present. |
| `docs/superpowers/specs/2026-05-08-okta-integration-handoff-to-hel.md` | Create | The 5/8 customer-facing deliverable for HeL, derived from spec §10. |

The boundary line: anything that touches the database goes through `orm/models.py` + `orm/functions.py` patterns. Anything that touches Streamlit auth state stays in `utils/okta_auth.py` and `utils/auth.py`. Tests mock `streamlit.secrets` and (for the auth handler) `streamlit.user` / `streamlit.login`.

---

## Task 1: Add `okta_sub` and `email` columns to the User model

**Files:**
- Modify: `orm/models.py:121-152` (User class column definitions)
- Test: `tests/utils/test_okta_auth.py` (new file, this task only adds the schema-shape test)

This is a schema change. Local mode is unaffected because the columns are nullable. The fixture infrastructure is added in Task 3; this task uses a minimal inline engine for the schema-shape assertion.

- [ ] **Step 1: Write the failing test**

Create `tests/utils/test_okta_auth.py` with this initial content:

```python
"""Unit tests for utils.okta_auth and the User-model OIDC columns.

Tests use an in-memory SQLite engine and only exercise pure helpers and the
sync_okta_user_to_db function against a scratch DB. No real OIDC traffic
flows in these tests; the full flow is validated manually against an Okta
Developer org per docs/superpowers/specs/2026-05-01-okta-oidc-integration-design.md §11.
"""

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/utils/test_okta_auth.py::test_user_model_has_okta_sub_and_email_columns -v`
Expected: FAIL — `AssertionError: User.okta_sub column missing`

- [ ] **Step 3: Add the columns to the User model**

Edit `orm/models.py`. Find the User class (currently around lines 121–152). After the `password` column (currently `password = Column(String(255), nullable=False)` at line 128) and before `created_at`, add:

```python
    # OIDC fields (NULL for local-only users; populated for users who
    # authenticate via Okta SSO). See
    # docs/superpowers/specs/2026-05-01-okta-oidc-integration-design.md §5.
    okta_sub = Column(String(255), nullable=True, unique=True)
    email = Column(String(320, collation="NOCASE"), nullable=True, unique=True)
```

Also relax the `password` column to nullable, because OIDC-provisioned users have no local password:

Change line 128 from:
```python
    password = Column(String(255), nullable=False)
```
to:
```python
    password = Column(String(255), nullable=True)  # NULL for OIDC-only users
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/utils/test_okta_auth.py::test_user_model_has_okta_sub_and_email_columns -v`
Expected: PASS.

- [ ] **Step 5: Run the existing test suite to verify nothing else broke**

Run: `uv run pytest -m "not milvus" -x`
Expected: All tests pass (or fail for reasons unrelated to this change). If a test fails because seed data assumes `password` is non-null, leave the assertion behavior — seeds still set passwords, so this should be fine.

- [ ] **Step 6: Commit**

```bash
git add orm/models.py tests/utils/test_okta_auth.py
git commit -m "feat(auth): add okta_sub and email columns to User model

Both columns are nullable so existing local-mode seeded users
keep working without backfill. okta_sub is the canonical Okta
identifier; email is the bootstrap key on first OIDC login.
SQLite COLLATE NOCASE makes the email unique constraint
case-insensitive natively.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Idempotent migration script for existing dev SQLite DBs

**Files:**
- Create: `scripts/migrate_add_okta_columns.py`

The repo uses `Base.metadata.create_all()` for fresh tables — that handles new installs. For existing developer DBs at `./pgDatabase/db.sqlite3`, we need an explicit `ALTER TABLE`. The script must be idempotent so running it twice is safe.

- [ ] **Step 1: Write the script**

Create `scripts/migrate_add_okta_columns.py`:

```python
"""One-time SQLite migration: add okta_sub and email columns to thrive_user.

Safe to run repeatedly — uses PRAGMA table_info to skip columns that
already exist. Run once on each developer's existing dev DB. New installs
do not need this script (Base.metadata.create_all handles them).

Usage:
    uv run python scripts/migrate_add_okta_columns.py
    uv run python scripts/migrate_add_okta_columns.py --db ./pgDatabase/db.sqlite3
"""

import argparse
import logging
import sqlite3
import sys

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

DEFAULT_DB = "./pgDatabase/db.sqlite3"


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    cursor = conn.execute(f"PRAGMA table_info({table})")
    return any(row[1] == column for row in cursor.fetchall())


def migrate(db_path: str) -> None:
    logger.info("Migrating %s", db_path)
    with sqlite3.connect(db_path) as conn:
        if _column_exists(conn, "thrive_user", "okta_sub"):
            logger.info("okta_sub already present — skipping")
        else:
            conn.execute("ALTER TABLE thrive_user ADD COLUMN okta_sub VARCHAR(255)")
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS ix_thrive_user_okta_sub ON thrive_user(okta_sub)")
            logger.info("Added okta_sub column and unique index")

        if _column_exists(conn, "thrive_user", "email"):
            logger.info("email already present — skipping")
        else:
            conn.execute("ALTER TABLE thrive_user ADD COLUMN email VARCHAR(320) COLLATE NOCASE")
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS ix_thrive_user_email ON thrive_user(email)")
            logger.info("Added email column and unique index")

        # SQLite cannot drop NOT NULL via ALTER TABLE. The original password
        # column was created NOT NULL; new fresh installs via Base.metadata.create_all
        # will use the relaxed nullability. Existing dev DBs continue to enforce
        # NOT NULL on password — acceptable because OIDC users won't be inserted
        # into pre-existing dev DBs without going through the new code path.
        # If you need OIDC users in an existing dev DB, supply a placeholder
        # password (e.g., "OIDC_USER") at insert time.
        conn.commit()
    logger.info("Migration complete")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=DEFAULT_DB, help="Path to SQLite DB (default: %(default)s)")
    args = parser.parse_args()
    try:
        migrate(args.db)
        return 0
    except Exception as exc:
        logger.error("Migration failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Verify script compiles and shows help**

Run: `uv run python scripts/migrate_add_okta_columns.py --help`
Expected: usage text appears, exit 0.

- [ ] **Step 3: Test idempotency on a temporary copy of the dev DB**

Run:
```bash
cp ./pgDatabase/db.sqlite3 /tmp/db_test.sqlite3
uv run python scripts/migrate_add_okta_columns.py --db /tmp/db_test.sqlite3
uv run python scripts/migrate_add_okta_columns.py --db /tmp/db_test.sqlite3
uv run python -c "import sqlite3; c=sqlite3.connect('/tmp/db_test.sqlite3'); print([r[1] for r in c.execute('PRAGMA table_info(thrive_user)')])"
```
Expected output (truncated): `[..., 'okta_sub', 'email']` (column names appear once even after running migration twice). The second migration run logs `okta_sub already present — skipping` and `email already present — skipping`.

- [ ] **Step 4: Run the migration on the real dev DB**

Run: `uv run python scripts/migrate_add_okta_columns.py`
Expected: log lines confirming the columns and indexes were added (or "already present" if you already ran it).

- [ ] **Step 5: Commit**

```bash
git add scripts/migrate_add_okta_columns.py
git commit -m "feat(auth): idempotent migration to add okta_sub and email to dev DBs

New installs are handled by Base.metadata.create_all; this script is for
existing developer SQLite databases that pre-date the column additions.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Add `in_memory_orm_session` test fixture

**Files:**
- Modify: `tests/conftest.py` (append to existing file)

Tests in Tasks 4–9 need an in-memory SQLAlchemy session with the schema applied and `UserRole`s seeded, plus a way to make production code (`orm.functions.create_user`, `orm.models.SessionLocal`) use that in-memory engine.

- [ ] **Step 1: Write the fixture and a smoke test**

Append to `tests/conftest.py`:

```python


# OIDC integration test fixture --------------------------------------------
@pytest.fixture
def in_memory_orm_session(monkeypatch):
    """In-memory SQLite ORM session with UserRoles seeded.

    Patches orm.models.SessionLocal and orm.models.engine so production code
    that imports SessionLocal (e.g. orm.functions.create_user) transparently
    uses this in-memory DB. Yields the SessionLocal class — call it to get
    a session. Tests typically use:

        with in_memory_orm_session() as session:
            session.query(User).all()
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from orm.models import Base, RoleTypeEnum, UserRole

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    # Seed the four UserRole rows so user_role_id FKs resolve.
    with TestingSessionLocal() as session:
        for role_name, description, role_enum in [
            ("Admin", "Administrator with full access", RoleTypeEnum.ADMIN),
            ("Doctor", "Physician with access to individual patient data", RoleTypeEnum.DOCTOR),
            ("Nurse", "Nurse with access to relevant patient data", RoleTypeEnum.NURSE),
            ("Patient", "Patient access only", RoleTypeEnum.PATIENT),
        ]:
            session.add(UserRole(role_name=role_name, description=description, role=role_enum))
        session.commit()

    monkeypatch.setattr("orm.models.SessionLocal", TestingSessionLocal)
    monkeypatch.setattr("orm.models.engine", engine)
    # orm.functions.SessionLocal is imported at module load time, so patch
    # there too. If a future caller does `from orm.models import SessionLocal`
    # in another module, add a similar monkeypatch line for that module.
    monkeypatch.setattr("orm.functions.SessionLocal", TestingSessionLocal)

    yield TestingSessionLocal

    engine.dispose()
```

Then add a smoke test at the end of `tests/utils/test_okta_auth.py`:

```python


def test_in_memory_orm_session_fixture_seeds_user_roles(in_memory_orm_session):
    """Smoke test: fixture should create the four UserRole rows."""
    from orm.models import UserRole

    with in_memory_orm_session() as session:
        names = {r.role_name for r in session.query(UserRole).all()}
        assert names == {"Admin", "Doctor", "Nurse", "Patient"}
```

- [ ] **Step 2: Run the smoke test to verify it passes**

Run: `uv run pytest tests/utils/test_okta_auth.py::test_in_memory_orm_session_fixture_seeds_user_roles -v`
Expected: PASS.

- [ ] **Step 3: Run the existing test suite**

Run: `uv run pytest -m "not milvus" -x`
Expected: All non-Milvus tests still pass. The new fixture is opt-in (only used when explicitly requested), so existing tests are not affected.

- [ ] **Step 4: Commit**

```bash
git add tests/conftest.py tests/utils/test_okta_auth.py
git commit -m "test: add in_memory_orm_session fixture for OIDC sync tests

Patches orm.models.SessionLocal and orm.functions.SessionLocal to point at
an in-memory SQLite DB with the four UserRole rows seeded, so tests of
sync_okta_user_to_db can run without touching the real dev DB.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Implement `role_id_from_groups`

**Files:**
- Create: `utils/okta_auth.py`
- Test: `tests/utils/test_okta_auth.py`

The pure mapping function: given a list of Okta group names (from the `groups` claim), return the `UserRole.id` for the highest-privilege match, or the default DOCTOR role id if no match.

- [ ] **Step 1: Write the failing tests**

Append to `tests/utils/test_okta_auth.py`:

```python


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
```

- [ ] **Step 2: Run tests — expect ImportError**

Run: `uv run pytest tests/utils/test_okta_auth.py -v -k role_id_from_groups`
Expected: FAIL — `ModuleNotFoundError: No module named 'utils.okta_auth'`.

- [ ] **Step 3: Create `utils/okta_auth.py` with `role_id_from_groups`**

Create `utils/okta_auth.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/utils/test_okta_auth.py -v -k role_id_from_groups`
Expected: all four `role_id_from_groups` tests PASS.

- [ ] **Step 5: Commit**

```bash
git add utils/okta_auth.py tests/utils/test_okta_auth.py
git commit -m "feat(auth): add role_id_from_groups for Okta group → role mapping

Highest-privilege group wins; default to DOCTOR if no match. The four
group names (thriveai-admin/doctor/nurse/patient) are the contract HeL
emits in the OIDC groups claim.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Implement `is_oidc_mode`

**Files:**
- Modify: `utils/okta_auth.py`
- Test: `tests/utils/test_okta_auth.py`

The dispatcher signal: returns True iff `secrets.toml` contains `[auth]` with `mode == "oidc"`. Pure function, easy to test by mocking `streamlit.secrets`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/utils/test_okta_auth.py`:

```python


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/utils/test_okta_auth.py -v -k is_oidc_mode`
Expected: FAIL — `ImportError: cannot import name 'is_oidc_mode'`.

- [ ] **Step 3: Add `is_oidc_mode` to `utils/okta_auth.py`**

Append to `utils/okta_auth.py`:

```python


def is_oidc_mode() -> bool:
    """True iff secrets.toml has [auth].mode == 'oidc'.

    Any other value, or a missing [auth] section, means local mode.
    """
    import streamlit as st

    auth_section = st.secrets.get("auth", {}) if hasattr(st, "secrets") else {}
    return auth_section.get("mode") == "oidc"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/utils/test_okta_auth.py -v -k is_oidc_mode`
Expected: all three tests PASS.

- [ ] **Step 5: Commit**

```bash
git add utils/okta_auth.py tests/utils/test_okta_auth.py
git commit -m "feat(auth): add is_oidc_mode dispatcher signal

Returns True iff secrets.toml has [auth].mode = 'oidc'. Used by
utils/auth.py to choose between OIDC and the existing local-cookie
auth path.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Implement `sync_okta_user_to_db`

**Files:**
- Modify: `utils/okta_auth.py`
- Test: `tests/utils/test_okta_auth.py`

The heart of the integration. Given a dict of OIDC claims (the same shape `st.user.to_dict()` returns), look up or create a `User` row, refresh the role from group claims, and return the row.

- [ ] **Step 1: Write the failing tests**

Append to `tests/utils/test_okta_auth.py`:

```python


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
    from orm.functions import create_user
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/utils/test_okta_auth.py -v -k sync_okta_user_to_db`
Expected: FAIL — `ImportError: cannot import name 'sync_okta_user_to_db'`.

- [ ] **Step 3: Implement `sync_okta_user_to_db`**

Append to `utils/okta_auth.py`:

```python


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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/utils/test_okta_auth.py -v -k sync_okta_user_to_db`
Expected: all five `sync_okta_user_to_db` tests PASS.

- [ ] **Step 5: Run the full file to confirm nothing regressed**

Run: `uv run pytest tests/utils/test_okta_auth.py -v`
Expected: All tests in the file PASS.

- [ ] **Step 6: Commit**

```bash
git add utils/okta_auth.py tests/utils/test_okta_auth.py
git commit -m "feat(auth): add sync_okta_user_to_db for JIT user provisioning

Resolves a User row from OIDC claims via okta_sub (canonical) or email
(bootstrap), or JIT-creates one. Role is refreshed from the groups claim
on every login per spec §6.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Implement `populate_session_state_from_user`

**Files:**
- Modify: `utils/okta_auth.py`
- Test: `tests/utils/test_okta_auth.py`

This helper translates a `User` row into the session-state shape used by the rest of the app. It's the bridge that lets downstream code (`extract_user_context_from_streamlit`, `_get_current_user_id`, `set_user_preferences_in_session_state`) work unchanged.

- [ ] **Step 1: Write the failing test**

Append to `tests/utils/test_okta_auth.py`:

```python


def test_populate_session_state_from_user_writes_expected_keys(in_memory_orm_session):
    """After population, session state mirrors what local-mode login produces."""
    import json
    from types import SimpleNamespace
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/utils/test_okta_auth.py::test_populate_session_state_from_user_writes_expected_keys -v`
Expected: FAIL — `ImportError: cannot import name 'populate_session_state_from_user'`.

- [ ] **Step 3: Implement `populate_session_state_from_user`**

Append to `utils/okta_auth.py`:

```python


def populate_session_state_from_user(user) -> None:
    """Mirror a User row into session state in the shape downstream code expects.

    After this returns, the app behaves identically to a local-mode login:
    cookies['user_id'], cookies['role_name'], session_state.user_role,
    session_state.username, and all preference flags are populated.
    """
    import json

    import streamlit as st

    from orm.functions import set_user_preferences_in_session_state

    st.session_state.cookies["user_id"] = json.dumps(user.id)
    st.session_state.cookies["role_name"] = user.role.role_name
    st.session_state.user_role = user.role.role.value
    st.session_state.username = f"{user.first_name} {user.last_name}".strip()

    # Reuse the existing preference loader; it reads cookies['user_id'] and
    # populates the same set of session-state keys local-mode login does.
    try:
        set_user_preferences_in_session_state()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("set_user_preferences_in_session_state failed: %s", exc)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/utils/test_okta_auth.py::test_populate_session_state_from_user_writes_expected_keys -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add utils/okta_auth.py tests/utils/test_okta_auth.py
git commit -m "feat(auth): add populate_session_state_from_user

Mirrors a User row into st.session_state in the same shape produced
by local-mode login, so downstream code (VannaService, RAG filters,
admin page gates) works unchanged regardless of which auth path ran.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Implement `handle_oidc_auth`, sidebar logout button, and `handle_oidc_logout`

**Files:**
- Modify: `utils/okta_auth.py`
- Test: `tests/utils/test_okta_auth.py`

These are the thin Streamlit-bound entry points. `handle_oidc_auth` checks `st.user.is_logged_in`, shows the SSO button if not, runs the sync + populate sequence if so, **and draws the sidebar welcome banner + logout button** (replacing what `_handle_local_auth` does in the local path). `handle_oidc_logout` clears VannaService cache, clears app session state, emits a redirect to `auth.post_logout_redirect_url`, then calls `st.logout()`.

**Note on the post-logout redirect:** Streamlit's `st.logout()` reruns the page; it does not natively support a custom post-logout URL. We emit a meta-refresh / JS redirect *before* calling `st.logout()` so the browser navigates to the Portal as the page is being torn down. This is best-effort and depends on browser timing. If the redirect doesn't fire, the user simply lands on the SSO button page and can navigate manually. Acceptable for the spike.

- [ ] **Step 1: Write the failing tests**

Append to `tests/utils/test_okta_auth.py`:

```python


def test_handle_oidc_auth_shows_login_button_when_not_logged_in(in_memory_orm_session):
    """If st.user.is_logged_in is False, render a login button and stop the page."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch

    from utils.okta_auth import handle_oidc_auth

    fake_user = SimpleNamespace(is_logged_in=False)
    button_mock = MagicMock(return_value=False)
    login_mock = MagicMock()
    stop_mock = MagicMock(side_effect=SystemExit)

    with patch("streamlit.user", fake_user), \
         patch("streamlit.button", button_mock), \
         patch("streamlit.login", login_mock), \
         patch("streamlit.stop", stop_mock), \
         patch("streamlit.title"), \
         patch("streamlit.markdown"):
        try:
            handle_oidc_auth()
        except SystemExit:
            pass

    button_mock.assert_called_once()  # SSO button rendered
    login_mock.assert_not_called()    # not clicked yet
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

    with patch("streamlit.user", fake_user), \
         patch("streamlit.button", button_mock), \
         patch("streamlit.login", login_mock), \
         patch("streamlit.stop", MagicMock(side_effect=SystemExit)), \
         patch("streamlit.title"), \
         patch("streamlit.markdown"):
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

    with patch("streamlit.user", fake_user), \
         patch("streamlit.session_state", fake_session_state), \
         patch("streamlit.sidebar", sidebar_mock), \
         patch("streamlit.title"), \
         patch("streamlit.button", button_mock), \
         patch("orm.functions.set_user_preferences_in_session_state", MagicMock()):
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

    with patch("streamlit.user", fake_user), \
         patch("streamlit.session_state", fake_session_state), \
         patch("streamlit.sidebar", sidebar_mock), \
         patch("streamlit.title"), \
         patch("streamlit.button", button_mock), \
         patch("utils.okta_auth.handle_oidc_logout", logout_mock), \
         patch("orm.functions.set_user_preferences_in_session_state", MagicMock()):
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
    fake_session_state["cookies"].get.return_value = '42'

    logout_mock = MagicMock()
    invalidate_mock = MagicMock()
    markdown_mock = MagicMock()

    with patch("streamlit.session_state", fake_session_state), \
         patch("streamlit.logout", logout_mock), \
         patch("streamlit.markdown", markdown_mock), \
         patch("streamlit.secrets", new={"auth": {"post_logout_redirect_url": "https://portal.example/"}}), \
         patch("utils.vanna_calls.VannaService.invalidate_cache_for_user", invalidate_mock):
        from utils.okta_auth import handle_oidc_logout

        handle_oidc_logout()

    invalidate_mock.assert_called_once_with("42", 1)
    logout_mock.assert_called_once()
    assert fake_session_state["messages"] == []
    assert fake_session_state["_vn_instance"] is None
    assert fake_session_state["selected_llm_provider"] is None
    assert fake_session_state["selected_llm_model"] is None
    # Redirect HTML was emitted before st.logout().
    redirect_call_found = any(
        "https://portal.example/" in str(call)
        for call in markdown_mock.call_args_list
    )
    assert redirect_call_found, "expected a redirect markdown to the post_logout_redirect_url"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/utils/test_okta_auth.py -v -k "handle_oidc"`
Expected: FAIL — `ImportError: cannot import name 'handle_oidc_auth'`.

- [ ] **Step 3: Implement `handle_oidc_auth` and `handle_oidc_logout`**

Append to `utils/okta_auth.py`:

```python


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
    claims = st.user.to_dict() if hasattr(st.user, "to_dict") else {
        "sub": getattr(st.user, "sub", None),
        "email": getattr(st.user, "email", None),
        "email_verified": getattr(st.user, "email_verified", None),
        "given_name": getattr(st.user, "given_name", ""),
        "family_name": getattr(st.user, "family_name", ""),
        "groups": getattr(st.user, "groups", []),
    }

    from orm.models import SessionLocal

    with SessionLocal() as session:
        user = sync_okta_user_to_db(claims, session)
        populate_session_state_from_user(user)
        # Cache attributes we need for the sidebar before the session closes.
        display_name = f"{user.first_name} {user.last_name}".strip()
        username = user.username
        user_id = user.id

    # Reuse existing login logging.
    try:
        from orm.logging_functions import log_login

        log_login(user_id=user_id, username=username, success=True)
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
    try:
        from utils.vanna_calls import VannaService

        user_id_str = st.session_state.cookies.get("user_id")
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

    # 3. Clear mirrored cookies.
    try:
        st.session_state.cookies["user_id"] = ""
        st.session_state.cookies["role_name"] = ""
        if hasattr(st.session_state.cookies, "save"):
            st.session_state.cookies.save()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to clear mirrored cookies on OIDC logout: %s", exc)

    # 4. Emit a meta-refresh redirect to the post-logout URL.
    redirect_url = (
        st.secrets.get("auth", {}).get("post_logout_redirect_url")
        if hasattr(st, "secrets")
        else None
    )
    if redirect_url:
        st.markdown(
            f'<meta http-equiv="refresh" content="0; url={redirect_url}">',
            unsafe_allow_html=True,
        )

    # 5. Drop Streamlit's auth cookie.
    st.logout()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/utils/test_okta_auth.py -v -k "handle_oidc"`
Expected: all four `handle_oidc_*` tests PASS.

- [ ] **Step 5: Run the full test file**

Run: `uv run pytest tests/utils/test_okta_auth.py -v`
Expected: every test in the file PASS.

- [ ] **Step 6: Commit**

```bash
git add utils/okta_auth.py tests/utils/test_okta_auth.py
git commit -m "feat(auth): add handle_oidc_auth and handle_oidc_logout entry points

handle_oidc_auth renders the SSO button on first hit and runs sync +
populate-session-state on every authenticated render. handle_oidc_logout
matches the existing local logout shape (clears VannaService cache,
session state, mirrored cookies) and then calls st.logout().

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Wire dispatcher into `utils/auth.py`

**Files:**
- Modify: `utils/auth.py` (entire file replaced; structure: dispatcher + private `_handle_local_auth`)
- Create: `tests/utils/test_auth_dispatcher.py`

`check_authenticate` becomes a dispatcher that calls `handle_oidc_auth` when `is_oidc_mode()` is true, or the existing local-cookie logic (renamed `_handle_local_auth`) otherwise. The local path is unchanged in behavior. The logout button inside the local path also dispatches to `handle_oidc_logout` if invoked from OIDC mode (defensive — it shouldn't fire there, since OIDC mode skips the local sidebar drawing — but documented as a no-op in OIDC mode anyway).

- [ ] **Step 1: Write the failing dispatcher tests**

Create `tests/utils/test_auth_dispatcher.py`:

```python
"""Verifies utils.auth.check_authenticate dispatches between OIDC and local."""

from unittest.mock import patch


def test_check_authenticate_routes_to_local_when_auth_section_absent():
    """Default behavior unchanged: no [auth] section → existing local flow runs."""
    import utils.auth as auth_module
    import utils.okta_auth as okta_module

    with patch("streamlit.secrets", new={}), \
         patch.object(auth_module, "_handle_local_auth") as local_mock, \
         patch.object(okta_module, "handle_oidc_auth") as oidc_mock:
        auth_module.check_authenticate()

    local_mock.assert_called_once()
    oidc_mock.assert_not_called()


def test_check_authenticate_routes_to_oidc_when_mode_is_oidc():
    """[auth].mode = 'oidc' routes to handle_oidc_auth."""
    import utils.auth as auth_module
    import utils.okta_auth as okta_module

    with patch("streamlit.secrets", new={"auth": {"mode": "oidc"}}), \
         patch.object(auth_module, "_handle_local_auth") as local_mock, \
         patch.object(okta_module, "handle_oidc_auth") as oidc_mock:
        auth_module.check_authenticate()

    oidc_mock.assert_called_once()
    local_mock.assert_not_called()


def test_check_authenticate_routes_to_local_when_mode_is_local():
    """[auth].mode = 'local' is an explicit fallback → local path."""
    import utils.auth as auth_module
    import utils.okta_auth as okta_module

    with patch("streamlit.secrets", new={"auth": {"mode": "local"}}), \
         patch.object(auth_module, "_handle_local_auth") as local_mock, \
         patch.object(okta_module, "handle_oidc_auth") as oidc_mock:
        auth_module.check_authenticate()

    local_mock.assert_called_once()
    oidc_mock.assert_not_called()
```

- [ ] **Step 2: Run the dispatcher tests to verify they fail**

Run: `uv run pytest tests/utils/test_auth_dispatcher.py -v`
Expected: FAIL — `AttributeError: module 'utils.auth' has no attribute '_handle_local_auth'`.

- [ ] **Step 3: Refactor `utils/auth.py` into a dispatcher**

Replace the entire contents of `utils/auth.py` with:

```python
import logging
from datetime import datetime, timedelta

import streamlit as st

import utils.okta_auth as okta_auth
from orm.functions import set_user_preferences_in_session_state, verify_user_credentials

logger = logging.getLogger(__name__)


def check_authenticate():
    """Dispatcher: route to OIDC handler or existing local-cookie path.

    The mode is determined by `[auth].mode` in secrets.toml. Absent or
    `mode = "local"` → existing username/password flow. `mode = "oidc"`
    → Streamlit-native OIDC via st.login() / st.user.

    Both paths leave session state in the same shape (cookies['user_id'],
    cookies['role_name'], session_state.user_role, session_state.username,
    plus all preference flags).

    Note: imports utils.okta_auth as a module (not the names) so test
    mocks of utils.okta_auth.handle_oidc_auth are visible at call time.
    """
    if okta_auth.is_oidc_mode():
        okta_auth.handle_oidc_auth()
    else:
        _handle_local_auth()


def _handle_local_auth():
    """Existing local-cookie auth path. Behavior unchanged from pre-OIDC."""
    try:
        user_id = st.session_state.cookies.get("user_id")
        expiry_date_str = st.session_state.cookies.get("expiry_date")
        if user_id and expiry_date_str:
            expiry_date = datetime.fromisoformat(expiry_date_str)
            user = set_user_preferences_in_session_state()

            # Ensure VannaService instance is cleared when switching users.
            if "_vn_instance" in st.session_state and st.session_state._vn_instance is not None:
                import json

                cached_user_id = getattr(st.session_state._vn_instance, "user_id", None)
                current_user_id = str(json.loads(user_id))
                if cached_user_id != current_user_id:
                    st.session_state._vn_instance = None

            if datetime.now() < expiry_date:
                cols = st.sidebar.columns([0.7, 0.3], vertical_alignment="bottom")
                with cols[0]:
                    st.title(f"Welcome {user.first_name} {user.last_name}")
                with cols[1]:
                    logout = st.button("Log Out")
                    if logout:
                        # Invalidate VannaService cache for this user.
                        from utils.vanna_calls import VannaService
                        import json

                        user_id_for_cache = json.loads(st.session_state.cookies.get("user_id"))
                        user_role = st.session_state.get("user_role")
                        if user_id_for_cache and user_role is not None:
                            VannaService.invalidate_cache_for_user(str(user_id_for_cache), user_role)

                        # Clear cookies.
                        st.session_state.cookies["user_id"] = ""
                        st.session_state.cookies["expiry_date"] = ""
                        st.session_state.cookies.save()

                        # Clear session state.
                        st.session_state.messages = []
                        st.session_state.selected_llm_provider = None
                        st.session_state.selected_llm_model = None
                        if "_vn_instance" in st.session_state:
                            st.session_state._vn_instance = None

                        st.rerun()
            else:
                _show_local_login()
        else:
            _show_local_login()
    except Exception as e:
        st.error(f"Error checking authentication: {e}")
        logger.error(f"Error checking authentication: {e}")


def _show_local_login():
    """Render the local username/password login form."""
    try:
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
        st.title("🔓 Log In")

        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button("Login", type="primary")

            if submit_button:
                if verify_user_credentials(username, password):
                    expiry_date = datetime.now() + timedelta(hours=8)
                    st.session_state.cookies["expiry_date"] = expiry_date.isoformat()
                    st.session_state.cookies.save()
                    st.rerun()
                else:
                    st.error("Incorrect username or password. Please try again.")
        st.stop()
    except Exception as e:
        st.error(f"Error showing login: {e}")
        logger.error(f"Error showing login: {e}")


# Backwards-compat alias: external imports of show_login still work.
show_login = _show_local_login
```

- [ ] **Step 4: Run the dispatcher tests to verify they pass**

Run: `uv run pytest tests/utils/test_auth_dispatcher.py -v`
Expected: all three dispatcher tests PASS.

- [ ] **Step 5: Run the full OIDC test suite**

Run: `uv run pytest tests/utils/test_okta_auth.py tests/utils/test_auth_dispatcher.py -v`
Expected: all tests PASS.

- [ ] **Step 6: Run the full repo test suite**

Run: `uv run pytest -m "not milvus" -x`
Expected: all non-Milvus tests PASS. Local mode is unchanged in behavior.

- [ ] **Step 7: Run lint and format**

Run:
```bash
uv run ruff check utils/auth.py utils/okta_auth.py tests/utils/test_okta_auth.py tests/utils/test_auth_dispatcher.py
uv run ruff format utils/auth.py utils/okta_auth.py tests/utils/test_okta_auth.py tests/utils/test_auth_dispatcher.py
```
Expected: zero lint errors after format.

- [ ] **Step 8: Commit**

```bash
git add utils/auth.py utils/okta_auth.py tests/utils/test_auth_dispatcher.py
git commit -m "feat(auth): wire OIDC/local dispatcher into check_authenticate

When secrets has [auth].mode = 'oidc', delegate to utils.okta_auth.
Otherwise run the existing local-cookie flow (now in _handle_local_auth).
Both paths leave session state in the same shape, so VannaService, RAG
filters, and admin page gates work unchanged.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: Manual end-to-end test against an Okta Developer org

**Files:**
- No code changes (manual verification step)
- Reference: `docs/superpowers/specs/2026-05-01-okta-oidc-integration-design.md` §11

This task validates the full OIDC flow end-to-end against a real Okta Developer org. It is intentionally manual — per the spec we did not build a mock OIDC server.

- [ ] **Step 1: Create an Okta Developer account**

Go to `https://developer.okta.com/signup/` and sign up. No credit card required. You receive a tenant URL like `https://dev-12345678.okta.com`.

- [ ] **Step 2: Create the OIDC web app**

In the Okta Admin Console:
1. Applications → Create App Integration → OIDC – OpenID Connect → Web Application → Next.
2. App integration name: `HEALTHeINTELLIGENCE Local Dev`.
3. Sign-in redirect URI: `http://localhost:8501/oauth2callback`.
4. Sign-out redirect URI: `http://localhost:8501/`.
5. Assignments: `Allow everyone in your organization to access` (we'll narrow per-group below).
6. Save. Note the `Client ID` and `Client secret`.

- [ ] **Step 3: Create the four groups**

Directory → Groups → Add Group, four times:
- `thriveai-admin`
- `thriveai-doctor`
- `thriveai-nurse`
- `thriveai-patient`

- [ ] **Step 4: Add the `groups` claim to the default authorization server**

Security → API → Authorization Servers → `default` → Claims → Add Claim:
- Name: `groups`
- Include in token type: `ID Token` / `Always`
- Value type: `Groups`
- Filter: `Matches regex` → `.*` (or `Starts with` → `thriveai-` to scope down)
- Include in: `Any scope` (or specifically the `openid` scope)
- Save.

- [ ] **Step 5: Create test users and assign groups**

Directory → People → Add Person, four times. Skip activation email; set passwords manually:
- `admin@test.local` → group `thriveai-admin`
- `doctor@test.local` → group `thriveai-doctor`
- `nurse@test.local` → group `thriveai-nurse`
- `patient@test.local` → group `thriveai-patient`

Then go back to Applications → your app → Assignments → Assign → assign all four groups (or assign each user individually).

- [ ] **Step 6: Wire `secrets.toml` for local OIDC mode**

Edit `.streamlit/secrets.toml` and add:

```toml
[auth]
mode = "oidc"
redirect_uri = "http://localhost:8501/oauth2callback"
cookie_secret = "REPLACE_WITH_A_RANDOM_32_PLUS_BYTE_STRING"
post_logout_redirect_url = "http://localhost:8501/"
client_id = "<Client ID from step 2>"
client_secret = "<Client secret from step 2>"
server_metadata_url = "https://dev-12345678.okta.com/oauth2/default/.well-known/openid-configuration"
client_kwargs = { scope = "openid email profile groups" }
```

Generate a `cookie_secret` with `python -c "import secrets; print(secrets.token_urlsafe(32))"` and substitute it.

(Single-provider form. To use the multi-provider form `st.login("okta")`, nest the client_id/client_secret/server_metadata_url under `[auth.okta]` and pass `"okta"` to `st.login()` in `utils/okta_auth.py`. Single-provider form is simpler for this spike.)

- [ ] **Step 7: Run the app**

```bash
uv run streamlit run app.py
```

Expected: the page renders the new SSO button "Sign in with HEALTHeCOMMUNITY (Okta)" instead of the username/password form.

- [ ] **Step 8: Test each role**

For each test user (`admin@test.local`, `doctor@test.local`, `nurse@test.local`, `patient@test.local`):

1. Click the SSO button.
2. Authenticate at Okta.
3. Verify you're redirected back to the chat UI.
4. Verify the right pages appear:
   - `admin@test.local` → "Admin Analytics" and "Feedback Dashboard" pages visible in nav.
   - Other users → only "Chat Bot" and "User Settings" visible.
5. Inspect `pgDatabase/db.sqlite3` and confirm a row exists in `thrive_user` with the expected `okta_sub`, `email`, and `user_role_id`.
6. Log out → confirm redirect to the configured post-logout URL.
7. Log in again → confirm the existing row is reused (no duplicate created).
8. Change one test user's group assignment in Okta (e.g. promote `doctor@test.local` to `thriveai-admin`). Log them in again. Confirm their `user_role_id` is updated and they now see the admin pages.

- [ ] **Step 9: Test the local-mode fallback**

Comment out the `[auth]` block in `secrets.toml` (or set `mode = "local"`). Restart Streamlit. Confirm the username/password form returns and an existing seeded user (`thriveai-re` / `password`) can still log in.

- [ ] **Step 10: Document any deviations**

If steps 8 or 9 surface unexpected behavior, capture the diff in `docs/superpowers/specs/2026-05-01-okta-oidc-integration-design.md` (or as a note attached to this plan), then fix the underlying code under TDD (write a failing test that reproduces the issue, fix, commit). No commit is needed for this step if everything works.

---

## Task 11: Write the HeL hand-off doc (5/8 deliverable)

**Files:**
- Create: `docs/superpowers/specs/2026-05-08-okta-integration-handoff-to-hel.md`

This is the customer-facing document that ThriveAI delivers to HeL by 5/8 per the meeting action items. It describes what HeL must configure on their Okta Classic side. It is derived from spec §10 and §13.

- [ ] **Step 1: Create the doc**

Create `docs/superpowers/specs/2026-05-08-okta-integration-handoff-to-hel.md`:

```markdown
# HEALTHeINTELLIGENCE — Okta OIDC Integration Requirements

**To:** HEALTHeLINK (HeL) — Robert Irvine, Ryan, Casey, Alyssa
**From:** ThriveAI — Rob Enderle
**Date:** 2026-05-08
**Status:** Requirements for HeL to configure the HEALTHeINTELLIGENCE app integration in HeC Portal Okta (Classic).

## 1. Summary

HEALTHeINTELLIGENCE integrates with the HEALTHeCOMMUNITY (HeC) Portal as a standard OIDC Relying Party. The app uses Streamlit's native OIDC support (Authlib under the hood). When a user clicks the Portal badge for HEALTHeINTELLIGENCE, they are redirected to the Okta authorization endpoint, authenticate (with Duo MFA on the Portal/Okta side), and are returned to the app with an ID token. The app reads identity, basic profile, and group membership from the ID token to drive role-based access.

The integration is **app-side**. No edge proxy or AWS ALB OIDC integration is required. The app handles the OIDC flow internally via `st.login()`.

## 2. What HeL must configure in Okta

| Field | Value |
|---|---|
| Application type | OIDC — Web Application |
| Grant type | Authorization Code (with PKCE) |
| Sign-in redirect URI | `https://<our-prod-host>/oauth2callback` (final hostname TBD by AWS team) and `http://localhost:8501/oauth2callback` (ThriveAI dev) |
| Sign-out redirect URI | `https://<our-prod-host>/` (or HeC Portal landing URL) |
| Required scopes | `openid`, `email`, `profile`, `groups` |
| Required claims in ID token | `sub`, `email`, `email_verified`, `given_name`, `family_name`, `groups` (custom) |
| Group claim format | JSON array of strings under claim name `groups` |
| Group names HeL must create and assign | `thriveai-admin`, `thriveai-doctor`, `thriveai-nurse`, `thriveai-patient` |
| Token endpoint authentication method | `client_secret_basic` |
| Logout behavior | App-side session clear only. We do NOT issue OIDC RP-initiated logout. User remains signed in to Okta and the Portal. |
| MFA | Enforced at Okta/Duo level. Transparent to the app. |
| Open prerequisite (HeL/AWS) | Production hostname + TLS cert. Current `IP:port` access cannot serve as an OIDC redirect target in production. |

## 3. What HeL must give back to ThriveAI

For each environment (dev/stage/prod, or however HeL splits them):

- Issuer URL — the `https://<okta>/oauth2/default` form, or whichever authorization server you wire up.
- `Client ID`
- `Client secret`
- Confirmation that the four `thriveai-*` groups exist and are emitted in the `groups` claim of the ID token.
- The exact production hostname HeL plans to use for HEALTHeINTELLIGENCE (so we register the right redirect URIs).

## 4. Authorization model on the app side

ThriveAI maps the `groups` claim to internal roles as follows:

| Okta group | App role | Capabilities |
|---|---|---|
| `thriveai-admin` | `Admin` | All pages, training data management, analytics, feedback dashboard. |
| `thriveai-doctor` | `Doctor` | Chat, user settings, full data view. |
| `thriveai-nurse` | `Nurse` | Chat, user settings, restricted data view. |
| `thriveai-patient` | `Patient` | Chat, user settings, most restricted data view. |

If a user is in multiple `thriveai-*` groups, the highest-privilege one wins. If a user is in **none** of these groups, they are treated as a `Doctor` by default (per stakeholder guidance to be permissive on go-live).

A user who authenticates successfully but is in no `thriveai-*` group still gets in. If you want to gate access entirely on group membership, tell us and we will change the default to "deny."

## 5. User provisioning

HEALTHeINTELLIGENCE uses **Just-in-Time (JIT) provisioning**. There is no SCIM connector. The first time a user authenticates via Okta, a row is created in our internal user table keyed off the `sub` claim, with email and name copied from the ID token. The user's role is refreshed from the `groups` claim on **every** login, so role changes in Okta take effect on the user's next login (no manual sync needed).

There is no "deactivate" path on the app side beyond Okta deactivation — if you remove a user in Okta, they cannot log in. Their data row in our app remains for audit purposes; we can purge on request.

## 6. Note on Okta Classic vs Identity Engine

ThriveAI's local validation environment is a free Okta Developer org. That org runs the Identity Engine. HeC Portal runs Okta Classic. The OIDC protocol surface our app touches is identical between the two engines, so no app changes are required. The **Okta admin UI workflows differ**, however — the steps to add a custom claim or configure an app are similar in spirit but laid out differently. Use HeL's existing Okta Classic playbooks; do not copy our screenshots.

## 7. Open questions to confirm by go-live

1. Is `groups` an acceptable claim name, or does HeL standardize on a different name (`group_membership`, `roles`, etc.)? If different, tell us and we will update one constant.
2. Will HeL provide a single tenant for all environments or separate tenants per environment? We can support either.
3. Should logout return the user to a specific Portal URL? Default in our config is the value HeL gives us; otherwise we'll redirect to HeC Portal's landing page.
4. Are there additional claims HeL wants us to enforce or display (e.g., `department`, `npi`, `organization_id`)? If so, list them and the field type, and we'll add them to the app.

## 8. Reference

The ThriveAI-side design and implementation plan are in:
- `docs/superpowers/specs/2026-05-01-okta-oidc-integration-design.md`
- `docs/superpowers/plans/2026-05-02-okta-oidc-integration.md`
```

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-05-08-okta-integration-handoff-to-hel.md
git commit -m "docs: HeL Okta integration requirements deliverable (5/8)

Customer-facing doc derived from the design spec. Covers what HeL must
configure in Okta, what they must give back to ThriveAI, the group →
role mapping, JIT provisioning behavior, and open questions to confirm
before go-live.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage check:**

| Spec section | Where implemented |
|---|---|
| §3 Decision 1 (auth modes by config) | Task 5 (`is_oidc_mode`), Task 9 (dispatcher) |
| §3 Decision 2 (JIT + group → role, default DOCTOR) | Task 4 (`role_id_from_groups`), Task 6 (`sync_okta_user_to_db`) |
| §3 Decision 3 (sub canonical, email bootstrap) | Task 6 |
| §3 Decision 4 (Streamlit native `st.login()`) | Task 8 (`handle_oidc_auth`) |
| §3 Decision 5 (Okta Developer org test) | Task 10 |
| §3 Decision 6 (logout returns to Portal, no RP-init) | Task 8 (`handle_oidc_logout`) |
| §4.1 module layout / §4.2 invariant | Tasks 4–9 collectively, Task 7 (`populate_session_state_from_user`) |
| §4.3 unchanged surfaces | Verified by Task 9 step 6 (full repo suite) |
| §5 schema changes | Tasks 1, 2 |
| §6 OIDC login flow | Tasks 6, 7, 8 |
| §7 group → role mapping | Task 4 |
| §8 logout flow | Task 8 |
| §9 secrets.toml shape | Task 10 step 6 |
| §10 deliverable for HeL | Task 11 |
| §11 Okta Developer walkthrough | Task 10 |
| §12 tests | Tasks 1, 4, 5, 6, 7, 8, 9 (all listed tests covered) |
| §13 risks (existing seeded users, cookie_secret rotation, MFA, refresh tokens) | Risks documented in Task 11 deliverable; cookie_secret guidance in Task 10 step 6 |
| §14 out of scope | Not implemented (correct) |
| §15 implementation sketch (file-level) | Tasks 1–9 + Task 11 |

**Placeholder scan:** No `TODO`, `TBD`, "implement later", "fill in details", or generic "add error handling" / "handle edge cases". `<our-prod-host>` and `<Client ID from step 2>` are intentional user-supplied values in the HeL deliverable doc and the Task 10 secrets snippet — these are content the engineer fills with real values they'll receive from HeL or Okta, not gaps in the plan.

**Type / signature consistency:**

| Symbol | Defined in | Referenced in |
|---|---|---|
| `is_oidc_mode() -> bool` | Task 5 | Task 9 dispatcher, Task 9 dispatcher tests |
| `role_id_from_groups(groups: Iterable[str], session) -> int` | Task 4 | Task 6 (`sync_okta_user_to_db`) |
| `sync_okta_user_to_db(claims: dict, session)` | Task 6 | Task 8 (`handle_oidc_auth`), tests in Task 7 |
| `populate_session_state_from_user(user)` | Task 7 | Task 8 (`handle_oidc_auth`) |
| `handle_oidc_auth()` | Task 8 | Task 9 dispatcher |
| `handle_oidc_logout()` | Task 8 | Wired into Task 8's `handle_oidc_auth` sidebar Log Out button. |
| `OKTA_GROUP_TO_ROLE`, `DEFAULT_ROLE_IF_NO_GROUP_MATCH` | Task 4 | Used by `role_id_from_groups` (Task 4); referenced in Task 11 deliverable |

All symbols match. The `handle_oidc_logout` import in `utils/auth.py` is harmless if unused — not a bug, just intentional surface.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-05-02-okta-oidc-integration.md`.** Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using `executing-plans`, batch execution with checkpoints.

Which approach?
