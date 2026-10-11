"""SQLAlchemy models for the PostgreSQL-compatible persistence schema."""

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import TypeDecorator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class UTCDateTime(TypeDecorator[datetime]):
    """Persist UTC and always return timezone-aware UTC datetimes."""

    impl = DateTime
    cache_ok = True

    def load_dialect_impl(self, dialect):
        return dialect.type_descriptor(DateTime(timezone=True))

    def process_bind_param(self, value: datetime | None, dialect):
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("UTCDateTime requires a timezone-aware value")
        normalized = value.astimezone(timezone.utc)
        if dialect.name == "sqlite":
            return normalized.replace(tzinfo=None)
        return normalized

    def process_result_value(self, value: datetime | None, dialect):
        del dialect
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)


JSON_DOCUMENT = JSON().with_variant(JSONB(), "postgresql")


class Base(DeclarativeBase):
    pass


class CaseModel(Base):
    __tablename__ = "cases"
    __table_args__ = (
        CheckConstraint("status IN ('active', 'closed', 'archived')", name="ck_cases_status"),
        Index("ix_cases_owner_created", "owner_subject", "created_at"),
    )
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    owner_subject: Mapped[str] = mapped_column(String(255), nullable=False)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    research_question: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active")
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utc_now, onupdate=utc_now)


class CaseDocumentModel(Base):
    __tablename__ = "case_documents"
    __table_args__ = (Index("ix_documents_case", "case_id", "uploaded_at"),)
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("cases.id", ondelete="RESTRICT"), nullable=False)
    owner_subject: Mapped[str] = mapped_column(String(255), nullable=False)
    safe_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    media_type: Mapped[str] = mapped_column(String(255), nullable=False)
    storage_key: Mapped[str] = mapped_column(String(1024), nullable=False, unique=True)
    source_metadata: Mapped[dict[str, Any]] = mapped_column(JSON_DOCUMENT, nullable=False, default=dict)
    uploaded_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utc_now)


class ResearchTaskModel(Base):
    __tablename__ = "research_tasks"
    __table_args__ = (
        CheckConstraint("status IN ('queued', 'running', 'partial', 'completed', 'failed')", name="ck_research_tasks_status"),
        UniqueConstraint("owner_subject", "request_fingerprint", name="uq_task_owner_fingerprint"),
        Index("ix_tasks_case_created", "case_id", "created_at"),
    )
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("cases.id", ondelete="RESTRICT"), nullable=False)
    owner_subject: Mapped[str] = mapped_column(String(255), nullable=False)
    request_fingerprint: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="queued")
    stages: Mapped[dict[str, str]] = mapped_column(JSON_DOCUMENT, nullable=False, default=dict)
    error_category: Mapped[str | None] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utc_now, onupdate=utc_now)
    completed_at: Mapped[datetime | None] = mapped_column(UTCDateTime())


class LegalReportModel(Base):
    __tablename__ = "legal_reports"
    __table_args__ = (
        CheckConstraint("state IN ('draft', 'partial', 'completed')", name="ck_legal_reports_state"),
        UniqueConstraint("task_id", name="uq_legal_reports_task"),
        Index("ix_reports_case_created", "case_id", "created_at"),
    )
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("cases.id", ondelete="RESTRICT"), nullable=False)
    task_id: Mapped[str] = mapped_column(ForeignKey("research_tasks.id", ondelete="RESTRICT"), nullable=False)
    owner_subject: Mapped[str] = mapped_column(String(255), nullable=False)
    state: Mapped[str] = mapped_column(String(32), nullable=False)
    report_payload: Mapped[dict[str, Any]] = mapped_column(JSON_DOCUMENT, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utc_now)


class AgentLogModel(Base):
    __tablename__ = "agent_logs"
    __table_args__ = (Index("ix_agent_logs_task_created", "task_id", "created_at"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("research_tasks.id", ondelete="RESTRICT"), nullable=False)
    owner_subject: Mapped[str] = mapped_column(String(255), nullable=False)
    agent_name: Mapped[str] = mapped_column(String(64), nullable=False)
    stage_name: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    error_category: Mapped[str | None] = mapped_column(String(128))
    safe_metadata: Mapped[dict[str, Any]] = mapped_column(JSON_DOCUMENT, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utc_now)
