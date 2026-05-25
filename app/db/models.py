"""SQLAlchemy ORM 模型定义"""

from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import Column, String, Text, Integer, BigInteger, DateTime, JSON, Float
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeMeta, declarative_base

Base: DeclarativeMeta = declarative_base()


class ConversationModel(Base):
    """会话表"""
    __tablename__ = "conversations"

    session_id = Column(String(255), primary_key=True)
    mode = Column(String(50), default="auto")
    title = Column(String(500), default="")
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class MessageModel(Base):
    """消息表"""
    __tablename__ = "messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    session_id = Column(String(255), nullable=False, index=True)
    role = Column(String(50), nullable=False)  # user / assistant / system
    content = Column(Text, nullable=False)
    token_count = Column(Integer, default=0)
    metadata_ = Column("metadata", JSON, default=dict)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class SessionSummaryModel(Base):
    """会话摘要表"""
    __tablename__ = "session_summaries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    session_id = Column(String(255), nullable=False, index=True)
    summary = Column(Text, nullable=False)
    summary_vector_id = Column(String(255), nullable=True)
    token_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class EvalRunModel(Base):
    """评估运行表"""
    __tablename__ = "eval_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    run_id = Column(String(100), nullable=False, unique=True, index=True)
    dataset_size = Column(Integer, default=0)
    metrics = Column(JSON, default=dict)
    report_path = Column(String(500), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class EvalItemModel(Base):
    """评估条目表"""
    __tablename__ = "eval_items"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    run_id = Column(String(100), nullable=False, index=True)
    question = Column(Text, default="")
    answer = Column(Text, default="")
    ground_truth = Column(Text, default="")
    contexts = Column(JSON, default=list)
    metrics = Column(JSON, default=dict)
    source = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
