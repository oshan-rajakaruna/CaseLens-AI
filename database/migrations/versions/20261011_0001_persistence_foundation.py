"""Create CaseLens persistence foundation.

Revision ID: 20261011_0001
Revises: None
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "20261011_0001"
down_revision = None
branch_labels = None
depends_on = None

json_document = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "cases",
        sa.Column("id", sa.String(128), primary_key=True),
        sa.Column("owner_subject", sa.String(255), nullable=False),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("description", sa.Text()),
        sa.Column("research_question", sa.Text()),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("status IN ('active', 'closed', 'archived')", name="ck_cases_status"),
    )
    op.create_index("ix_cases_owner_created", "cases", ["owner_subject", "created_at"])
    op.create_table(
        "case_documents",
        sa.Column("id", sa.String(128), primary_key=True),
        sa.Column("case_id", sa.String(128), sa.ForeignKey("cases.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("owner_subject", sa.String(255), nullable=False),
        sa.Column("safe_filename", sa.String(255), nullable=False),
        sa.Column("media_type", sa.String(255), nullable=False),
        sa.Column("storage_key", sa.String(1024), nullable=False, unique=True),
        sa.Column("source_metadata", json_document, nullable=False),
        sa.Column("uploaded_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_documents_case", "case_documents", ["case_id", "uploaded_at"])
    op.create_table(
        "research_tasks",
        sa.Column("id", sa.String(128), primary_key=True),
        sa.Column("case_id", sa.String(128), sa.ForeignKey("cases.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("owner_subject", sa.String(255), nullable=False),
        sa.Column("request_fingerprint", sa.String(128), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("stages", json_document, nullable=False),
        sa.Column("error_category", sa.String(128)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint("status IN ('queued', 'running', 'partial', 'completed', 'failed')", name="ck_research_tasks_status"),
        sa.UniqueConstraint("owner_subject", "request_fingerprint", name="uq_task_owner_fingerprint"),
    )
    op.create_index("ix_tasks_case_created", "research_tasks", ["case_id", "created_at"])
    op.create_table(
        "legal_reports",
        sa.Column("id", sa.String(128), primary_key=True),
        sa.Column("case_id", sa.String(128), sa.ForeignKey("cases.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("task_id", sa.String(128), sa.ForeignKey("research_tasks.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("owner_subject", sa.String(255), nullable=False),
        sa.Column("state", sa.String(32), nullable=False),
        sa.Column("report_payload", json_document, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("state IN ('draft', 'partial', 'completed')", name="ck_legal_reports_state"),
        sa.UniqueConstraint("task_id", name="uq_legal_reports_task"),
    )
    op.create_index("ix_reports_case_created", "legal_reports", ["case_id", "created_at"])
    op.create_table(
        "agent_logs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("task_id", sa.String(128), sa.ForeignKey("research_tasks.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("owner_subject", sa.String(255), nullable=False),
        sa.Column("agent_name", sa.String(64), nullable=False),
        sa.Column("stage_name", sa.String(64), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("error_category", sa.String(128)),
        sa.Column("safe_metadata", json_document, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_agent_logs_task_created", "agent_logs", ["task_id", "created_at"])


def downgrade() -> None:
    op.drop_table("agent_logs")
    op.drop_table("legal_reports")
    op.drop_table("research_tasks")
    op.drop_table("case_documents")
    op.drop_table("cases")
