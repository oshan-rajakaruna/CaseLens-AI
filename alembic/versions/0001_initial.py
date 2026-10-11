"""initial CaseLens records
Revision ID: 0001_initial
"""
from alembic import op
import sqlalchemy as sa
revision="0001_initial"; down_revision=None; branch_labels=None; depends_on=None
def upgrade():
    op.create_table("cases", sa.Column("id", sa.String(36), primary_key=True), sa.Column("owner_id", sa.String(128), nullable=False), sa.Column("title", sa.String(500), nullable=False), sa.Column("description", sa.Text()), sa.Column("jurisdiction", sa.String(200)), sa.Column("status", sa.String(32), nullable=False), sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()), sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()))
    op.create_index("ix_cases_owner_id", "cases", ["owner_id"])
    op.create_table("documents", sa.Column("id", sa.String(36), primary_key=True), sa.Column("case_id", sa.String(36), sa.ForeignKey("cases.id", ondelete="CASCADE"), nullable=False), sa.Column("owner_id", sa.String(128), nullable=False), sa.Column("original_name", sa.String(255), nullable=False), sa.Column("media_type", sa.String(100), nullable=False), sa.Column("storage_key", sa.String(255), nullable=False, unique=True), sa.Column("size_bytes", sa.Integer(), nullable=False), sa.Column("extraction_status", sa.String(32), nullable=False), sa.Column("extracted_text", sa.Text()), sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()))
    op.create_table("research_jobs", sa.Column("id", sa.String(36), primary_key=True), sa.Column("case_id", sa.String(36), sa.ForeignKey("cases.id", ondelete="CASCADE"), nullable=False), sa.Column("owner_id", sa.String(128), nullable=False), sa.Column("status", sa.String(32), nullable=False), sa.Column("result", sa.JSON()), sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()), sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()))
def downgrade():
    op.drop_table("research_jobs"); op.drop_table("documents"); op.drop_index("ix_cases_owner_id", table_name="cases"); op.drop_table("cases")
