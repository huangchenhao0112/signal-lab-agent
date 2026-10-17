from __future__ import annotations

from sqlalchemy import (
    delete,
    select,
)

from sqlalchemy.orm import Session

from app.database.models import (
    AgentSession,
    AgentMessage,
)


def get_agent_session(
    db: Session,
    session_id: str,
) -> AgentSession | None:
    """
    根据session_id查询会话。
    """

    statement = (
        select(
            AgentSession
        )
        .where(
            AgentSession.session_id
            == session_id
        )
    )

    result = db.execute(
        statement
    )

    return (
        result.scalar_one_or_none()
    )


def create_agent_session(
    db: Session,
    session_id: str,
    title: str | None = None,
) -> AgentSession:
    """
    创建新的Agent Session。
    """

    record = AgentSession(
        session_id=session_id,
        title=title,
    )

    db.add(
        record
    )

    db.commit()

    db.refresh(
        record
    )

    return record


def get_or_create_agent_session(
    db: Session,
    session_id: str,
    title: str | None = None,
) -> tuple[
    AgentSession,
    bool,
]:
    """
    查询Session。

    不存在：
        创建

    返回：
        (record, created)
    """

    record = get_agent_session(
        db,
        session_id,
    )

    if record is not None:
        return (
            record,
            False,
        )

    record = create_agent_session(
        db=db,
        session_id=session_id,
        title=title,
    )

    return (
        record,
        True,
    )


def update_agent_session_title(
    db: Session,
    session_id: str,
    title: str,
) -> AgentSession | None:
    """
    修改Session标题。
    """

    record = get_agent_session(
        db,
        session_id,
    )

    if record is None:
        return None

    record.title = title

    db.commit()

    db.refresh(
        record
    )

    return record


def touch_agent_session(
    db: Session,
    session_id: str,
) -> AgentSession | None:
    """
    更新Session更新时间。

    SQLAlchemy的onupdate通常会在UPDATE发生时更新updated_at。

    这里通过重新赋值title触发更新。
    """

    record = get_agent_session(
        db,
        session_id,
    )

    if record is None:
        return None

    record.title = record.title

    db.commit()

    db.refresh(
        record
    )

    return record


def delete_agent_session(
    db: Session,
    session_id: str,
) -> bool:
    """
    删除Session元数据。
    """

    statement = (
        delete(
            AgentSession
        )
        .where(
            AgentSession.session_id
            == session_id
        )
    )

    result = db.execute(
        statement
    )

    db.commit()

    return (
        result.rowcount or 0
    ) > 0

def save_agent_message(
    db: Session,
    session_id: str,
    role: str,
    content: str,
    trace_id: str | None = None,
) -> AgentMessage:

    record = AgentMessage(
        session_id=session_id,
        role=role,
        content=content,
        trace_id=trace_id,
    )

    db.add(record)
    db.commit()
    db.refresh(record)

    return record

def list_agent_messages(
    db: Session,
    session_id: str,
) -> list[AgentMessage]:

    statement = (
        select(AgentMessage)
        .where(
            AgentMessage.session_id
            == session_id
        )
        .order_by(
            AgentMessage.created_at.asc()
        )
    )

    result = db.execute(statement)

    return list(
        result.scalars().all()
    )