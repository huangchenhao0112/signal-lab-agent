#第一张真实业务表
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    DateTime,
    String,
    func,
)

from sqlalchemy.orm import (
    Mapped,
    mapped_column,
)

from app.database.mysql import Base
from sqlalchemy import (
    DateTime,
    String,
    Text,
    func,
)

class AgentSession(
    Base
):
    """
    Agent会话元数据表。

    注意：
    这里暂时不保存完整聊天历史。

    Redis负责：
        recent messages
        conversation summary

    MySQL负责：
        session元数据
        title
        创建时间
        更新时间
    """

    __tablename__ = (
        "agent_sessions"
    )

    id: Mapped[int] = mapped_column(
        primary_key=True,
        autoincrement=True,
    )

    session_id: Mapped[str] = (
        mapped_column(
            String(64),
            unique=True,
            nullable=False,
            index=True,
        )
    )

    title: Mapped[str | None] = (
        mapped_column(
            String(255),
            nullable=True,
        )
    )

    created_at: Mapped[datetime] = (
        mapped_column(
            DateTime,
            nullable=False,
            server_default=func.now(),
        )
    )

    updated_at: Mapped[datetime] = (
        mapped_column(
            DateTime,
            nullable=False,
            server_default=func.now(),
            onupdate=func.now(),
        )
    )

class AgentMessage(
    Base
):
    """
    Agent完整聊天记录。

    MySQL负责长期保存，
    Redis只保留当前活跃上下文。
    """

    __tablename__ = "agent_messages"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        autoincrement=True,
    )

    session_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
    )

    role: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    trace_id: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        index=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.now(),
    )
