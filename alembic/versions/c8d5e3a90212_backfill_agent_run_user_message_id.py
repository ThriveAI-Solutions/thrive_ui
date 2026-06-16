"""backfill thrive_agent_run.user_message_id

A placeholder in ``agent/deps_builder.build_agent_deps`` (introduced in
revision ``401a2a3``) left ``AgentDeps.user_message_id`` hardcoded to None,
so every ``AgentRun`` row written since then carries NULL in
``user_message_id``. The per-query audit view joins
``AgentRun.user_message_id == Message.id`` (see
``orm/logging_functions.py``) and any miss falls through to the legacy half
of the UNION ALL, mislabelling Pipeline/Scope/Tool as
"legacy" / "Legacy/Unknown" / "(legacy SQL)".

This data-only migration retroactively wires each ``AgentRun`` with a NULL
``user_message_id`` to the latest USER ``Message`` for the same
``(user_id, question)`` whose ``created_at`` is at-or-before the run's own
``created_at``. The USER message is always written by ``set_question`` in
``utils/chat_bot_helper.py`` immediately before the rerun lands in
``agent/runtime.run_agentic_flow`` — so the gap is normally sub-second; we
allow up to one hour as a safety margin without bounding the lookback
unrealistically.

Idempotent: re-running upgrade only touches rows whose ``user_message_id``
is still NULL, so a previously-backfilled row is never reassigned. Already
linked runs (e.g. once the production fix lands and new rows write a real
id at insert time) are untouched.

Downgrade is a no-op — clearing ``user_message_id`` back to NULL would
destroy correct linkage data and re-introduce the bug for historical rows.

Revision ID: c8d5e3a90212
Revises: 9a4c12e7b001
Create Date: 2026-06-15 21:30:00.000000

"""

from typing import Sequence, Union

from alembic import op


revision: str = "c8d5e3a90212"
down_revision: Union[str, Sequence[str], None] = "9a4c12e7b001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# Match an AgentRun to the most recent USER message with the same content,
# owned by the same user, written at-or-before the run's created_at and
# within a 1-hour lookback. Correlated subquery for SQLite compatibility.
_BACKFILL_SQL = """
UPDATE thrive_agent_run
SET user_message_id = (
    SELECT m.id
    FROM thrive_message AS m
    WHERE m.user_id = thrive_agent_run.user_id
      AND m.role = 'user'
      AND m.content = thrive_agent_run.question
      AND m.created_at <= thrive_agent_run.created_at
      AND m.created_at >= datetime(thrive_agent_run.created_at, '-1 hour')
    ORDER BY m.created_at DESC, m.id DESC
    LIMIT 1
)
WHERE thrive_agent_run.user_message_id IS NULL
  AND thrive_agent_run.question IS NOT NULL
"""


def upgrade() -> None:
    op.execute(_BACKFILL_SQL)


def downgrade() -> None:
    # Intentional no-op: clearing the backfilled values would re-introduce
    # the audit-view legacy-mislabel bug for every historical row.
    pass
