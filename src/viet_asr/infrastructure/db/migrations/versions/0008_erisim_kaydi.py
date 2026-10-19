"""Erişim kaydı (KVKK): kim, ne zaman, hangi işte ne yaptı.

Revision ID: 0008
Revises: 0007
Create Date: 2026-10-19
"""
import sqlalchemy as sa
from alembic import op

from viet_asr.infrastructure.db.models import UtcDateTime

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "audit_log",
        sa.Column("id", sa.BigInteger().with_variant(sa.Integer(), "sqlite"), primary_key=True,
                  autoincrement=True),
        sa.Column("at", UtcDateTime(), nullable=False),
        sa.Column("actor", sa.Text(), nullable=False),
        sa.Column("action", sa.Text(), nullable=False),
        sa.Column("job_id", sa.Text(), nullable=True),
        sa.Column("detail", sa.Text(), nullable=True),
        sa.Column("ip", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_audit_log")),
    )
    op.create_index("ix_audit_log_at", "audit_log", ["at"])
    op.create_index("ix_audit_log_job_id", "audit_log", ["job_id"])


def downgrade() -> None:
    op.drop_index("ix_audit_log_job_id", table_name="audit_log")
    op.drop_index("ix_audit_log_at", table_name="audit_log")
    op.drop_table("audit_log")
