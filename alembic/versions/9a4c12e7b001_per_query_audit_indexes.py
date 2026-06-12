"""add composite (source_id, run_id) index on thrive_agent_patient_access

Epic #190 (audit log parity for the agentic flow). Phase 1's per-query
audit data-access layer joins ``AgentPatientAccess`` by ``source_id`` to
support the By-Patient pivot ("every query that touched this patient")
and decorates each row with the run's patient touches. The existing
single-column ``ix_thrive_agent_patient_access_source`` covers the
``source_id`` lookup but a composite index on ``(source_id, run_id)``
covers the common pivot pattern (find runs that touched a patient, then
deduplicate by run) without a separate join.

The index is hygiene; safe to leave in place on rollback. Migration is
symmetric (create / drop).

Revision ID: 9a4c12e7b001
Revises: 7b3a1f0c92d4
Create Date: 2026-06-12 15:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
from sqlalchemy import inspect

# revision identifiers, used by Alembic.
revision: str = "9a4c12e7b001"
down_revision: Union[str, Sequence[str], None] = "7b3a1f0c92d4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_INDEX_NAME = "ix_thrive_agent_patient_access_source_run"
_TABLE_NAME = "thrive_agent_patient_access"
_COLUMNS = ["source_id", "run_id"]


def upgrade() -> None:
    """Upgrade schema."""
    bind = op.get_bind()
    inspector = inspect(bind)
    existing = {ix["name"] for ix in inspector.get_indexes(_TABLE_NAME)}
    if _INDEX_NAME not in existing:
        op.create_index(_INDEX_NAME, _TABLE_NAME, _COLUMNS)


def downgrade() -> None:
    """Downgrade schema."""
    bind = op.get_bind()
    inspector = inspect(bind)
    existing = {ix["name"] for ix in inspector.get_indexes(_TABLE_NAME)}
    if _INDEX_NAME in existing:
        op.drop_index(_INDEX_NAME, table_name=_TABLE_NAME)
