"""require user email, organization, user_role_id (Epic #179)

Backfills any NULL ``email``, ``organization``, or ``user_role_id`` rows in
``thrive_user``, then flips those three columns to ``NOT NULL`` via
``op.batch_alter_table`` (SQLite-compatible per CLAUDE.md schema-change
guidance).

Backfill strategy — all sentinel values are intentionally traceable so a
post-migration admin sweep can find and replace them with real data:

- ``email``  → ``"<missing-email-{id}@unknown.local>"`` (unique per row,
  satisfies the case-insensitive unique constraint and the lightweight
  regex in ``orm.functions._is_valid_email``).
- ``organization`` → ``"Unknown"``.
- ``user_role_id`` → the PATIENT role id (lowest privilege, matches the
  OIDC JIT fallback so behavior is consistent across paths).

Every backfilled row is printed to stdout with the prefix
``BACKFILL MANIFEST:`` so the admin running the migration can capture the
list from the Alembic output and review post-migration.

``downgrade()`` only relaxes the ``NOT NULL`` flags; the sentinel values
stay (their format makes them findable via ``LIKE '%@unknown.local%'`` or
``organization = 'Unknown'``).

Revision ID: 7b3a1f0c92d4
Revises: 188ab391e291
Create Date: 2026-06-11 12:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
from sqlalchemy import inspect, text

# revision identifiers, used by Alembic.
revision: str = "7b3a1f0c92d4"
down_revision: Union[str, Sequence[str], None] = "188ab391e291"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_TABLE = "thrive_user"
_ROLE_TABLE = "thrive_user_role"


def _patient_role_id(bind) -> int:
    """Resolve PATIENT role id, or fall back to MIN(id) if the row is missing.

    The four canonical UserRole rows are seeded by ``orm.models.seed_initial_data``,
    so in production this always returns the PATIENT id. The fallback only
    triggers for malformed DBs (no roles seeded yet) — picking the lowest
    existing id keeps the FK valid; admin review will catch it.
    """
    row = bind.execute(
        text(f"SELECT id FROM {_ROLE_TABLE} WHERE role = :role"),
        {"role": "PATIENT"},
    ).first()
    if row is not None:
        return int(row[0])
    fallback = bind.execute(text(f"SELECT MIN(id) FROM {_ROLE_TABLE}")).scalar()
    if fallback is None:
        raise RuntimeError(
            f"Cannot backfill user_role_id — {_ROLE_TABLE} has no rows. "
            "Seed UserRole rows before running this migration."
        )
    print(f"BACKFILL MANIFEST: WARN PATIENT role row missing; falling back to UserRole.id={fallback}")
    return int(fallback)


def _log_manifest(user_id: int, field: str, old, new) -> None:
    """Emit one line per backfilled value so admins can audit the migration run."""
    print(f"BACKFILL MANIFEST: user_id={user_id} field={field} old={old!r} new={new!r}")


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)

    user_cols = {c["name"]: c for c in inspector.get_columns(_TABLE)}
    if "email" not in user_cols or "organization" not in user_cols or "user_role_id" not in user_cols:
        # Fresh schema (e.g. baseline-only test DB without the org column).
        # Nothing to backfill; downstream ALTER will create the columns as
        # NOT NULL via the model definition on next create_all. Skip gracefully.
        return

    # ── 1. Backfill user_role_id ─────────────────────────────────────────
    null_role_rows = bind.execute(text(f"SELECT id, user_role_id FROM {_TABLE} WHERE user_role_id IS NULL")).fetchall()
    if null_role_rows:
        patient_id = _patient_role_id(bind)
        for row in null_role_rows:
            _log_manifest(row[0], "user_role_id", None, patient_id)
        bind.execute(
            text(f"UPDATE {_TABLE} SET user_role_id = :rid WHERE user_role_id IS NULL"),
            {"rid": patient_id},
        )

    # ── 2. Backfill organization ─────────────────────────────────────────
    null_org_rows = bind.execute(
        text(f"SELECT id FROM {_TABLE} WHERE organization IS NULL OR TRIM(organization) = ''")
    ).fetchall()
    if null_org_rows:
        for row in null_org_rows:
            _log_manifest(row[0], "organization", None, "Unknown")
        bind.execute(
            text(f"UPDATE {_TABLE} SET organization = 'Unknown' WHERE organization IS NULL OR TRIM(organization) = ''")
        )

    # ── 3. Backfill email ────────────────────────────────────────────────
    # Each row gets a unique synthetic email so the case-insensitive unique
    # constraint is satisfied even when multiple rows are backfilled at once.
    null_email_rows = bind.execute(text(f"SELECT id FROM {_TABLE} WHERE email IS NULL OR TRIM(email) = ''")).fetchall()
    for row in null_email_rows:
        synthetic = f"<missing-email-{row[0]}@unknown.local>"
        _log_manifest(row[0], "email", None, synthetic)
        bind.execute(
            text(f"UPDATE {_TABLE} SET email = :email WHERE id = :uid"),
            {"email": synthetic, "uid": row[0]},
        )

    # ── 4. Apply NOT NULL via batch ALTER (SQLite-safe) ──────────────────
    with op.batch_alter_table(_TABLE, schema=None) as batch_op:
        batch_op.alter_column("email", existing_type=None, nullable=False)
        batch_op.alter_column("organization", existing_type=None, nullable=False)
        batch_op.alter_column("user_role_id", existing_type=None, nullable=False)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)

    user_cols = {c["name"]: c for c in inspector.get_columns(_TABLE)}
    if "email" not in user_cols or "organization" not in user_cols or "user_role_id" not in user_cols:
        return

    with op.batch_alter_table(_TABLE, schema=None) as batch_op:
        batch_op.alter_column("email", existing_type=None, nullable=True)
        batch_op.alter_column("organization", existing_type=None, nullable=True)
        batch_op.alter_column("user_role_id", existing_type=None, nullable=True)
