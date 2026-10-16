"""Konuşmacı profili (ses perdesi, kaba cinsiyet) ve satır içi cümle sınırları.

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-16
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("jobs") as b:
        b.add_column(sa.Column("speaker_info", sa.JSON().with_variant(postgresql.JSONB(), "postgresql"),
                               nullable=True))
    with op.batch_alter_table("utterances") as b:
        b.add_column(sa.Column("breaks", sa.Text(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("utterances") as b:
        b.drop_column("breaks")
    with op.batch_alter_table("jobs") as b:
        b.drop_column("speaker_info")
