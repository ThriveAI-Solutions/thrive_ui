"""vocab tables for search_codes

Revision ID: 27f6a69b2350
Revises: c233fb500001
Create Date: 2026-07-06 22:09:52.055617

Adds the 5 search_codes vocabulary tables (ported from chiron's
core/db/models.py — spec 2026-07-03). Global reference data with no
per-user/tenant scoping; loaded only by a later ingest script. Tables are
empty after this migration — zero behavior change.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


# revision identifiers, used by Alembic.
revision: str = "27f6a69b2350"
down_revision: Union[str, Sequence[str], None] = "c233fb500001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    bind = op.get_bind()
    inspector = inspect(bind)
    existing_tables = set(inspector.get_table_names())

    if "vocab_code_sets" not in existing_tables:
        op.create_table(
            "vocab_code_sets",
            sa.Column("set_id", sa.String(length=128), nullable=False),
            sa.Column("name", sa.Text(), nullable=False),
            sa.Column("source", sa.String(length=32), nullable=False),
            sa.Column("source_version", sa.String(length=64), nullable=True),
            sa.PrimaryKeyConstraint("set_id"),
        )

    if "vocab_codes" not in existing_tables:
        op.create_table(
            "vocab_codes",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("vocabulary", sa.String(length=16), nullable=False),
            sa.Column("code", sa.String(length=64), nullable=False),
            sa.Column("code_norm", sa.String(length=64), nullable=False),
            sa.Column("display", sa.Text(), nullable=False),
            sa.Column("is_active", sa.Boolean(), nullable=False),
            sa.Column("source_version", sa.String(length=64), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_vocab_codes_vocab_norm", "vocab_codes", ["vocabulary", "code_norm"], unique=True)

    if "vocab_code_set_members" not in existing_tables:
        op.create_table(
            "vocab_code_set_members",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("set_id", sa.String(length=128), nullable=False),
            sa.Column("code_id", sa.Integer(), nullable=False),
            sa.ForeignKeyConstraint(["code_id"], ["vocab_codes.id"], ondelete="CASCADE"),
            sa.ForeignKeyConstraint(["set_id"], ["vocab_code_sets.set_id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_vocab_set_members_code_id", "vocab_code_set_members", ["code_id"])
        op.create_index(
            "ix_vocab_set_members_set_code",
            "vocab_code_set_members",
            ["set_id", "code_id"],
            unique=True,
        )

    if "vocab_set_synonyms" not in existing_tables:
        op.create_table(
            "vocab_set_synonyms",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("set_id", sa.String(length=128), nullable=False),
            sa.Column("term", sa.Text(), nullable=False),
            sa.Column("term_norm", sa.Text(), nullable=False),
            sa.ForeignKeyConstraint(["set_id"], ["vocab_code_sets.set_id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_vocab_set_synonyms_term_norm", "vocab_set_synonyms", ["term_norm"])

    if "vocab_synonyms" not in existing_tables:
        op.create_table(
            "vocab_synonyms",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("code_id", sa.Integer(), nullable=False),
            sa.Column("term", sa.Text(), nullable=False),
            sa.Column("term_norm", sa.Text(), nullable=False),
            sa.Column("source", sa.String(length=32), nullable=True),
            sa.Column("is_lay", sa.Boolean(), nullable=False),
            sa.ForeignKeyConstraint(["code_id"], ["vocab_codes.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_vocab_synonyms_code_id", "vocab_synonyms", ["code_id"])
        op.create_index("ix_vocab_synonyms_term_norm", "vocab_synonyms", ["term_norm"])


def downgrade() -> None:
    """Downgrade schema (children first)."""
    op.drop_table("vocab_synonyms")
    op.drop_table("vocab_set_synonyms")
    op.drop_table("vocab_code_set_members")
    op.drop_table("vocab_codes")
    op.drop_table("vocab_code_sets")
