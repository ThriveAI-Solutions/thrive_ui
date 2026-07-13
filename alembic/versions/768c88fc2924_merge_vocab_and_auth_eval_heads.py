"""merge vocab and auth-eval heads

Revision ID: 768c88fc2924
Revises: 27f6a69b2350, d4f27a91c8b3
Create Date: 2026-07-12 21:30:22.602023

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '768c88fc2924'
down_revision: Union[str, Sequence[str], None] = ('27f6a69b2350', 'd4f27a91c8b3')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
