"""İş ilerlemesi: işlenirken aşama adı ve yaklaşık yüzde (arayüz ve GET /jobs/{id}).

Revision ID: 0007
Revises: 0006
Create Date: 2026-10-19
"""
import sqlalchemy as sa
from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("jobs") as b:
        b.add_column(sa.Column("stage", sa.Text(), nullable=True))
        b.add_column(sa.Column("progress", sa.Float(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("jobs") as b:
        b.drop_column("progress")
        b.drop_column("stage")
