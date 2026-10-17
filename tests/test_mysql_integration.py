from __future__ import annotations

import uuid

from app.database.mysql import SessionLocal
from app.database.models import AgentSession, AgentMessage
import pytest

pytestmark = pytest.mark.integration

def test_mysql_session_and_message_crud():
    """
    验证 MySQL：
    1. 能创建 session
    2. 能写入 message
    3. 能查询
    4. 能删除
    """

    db = SessionLocal()

    session_id = f"pytest-{uuid.uuid4()}"

    try:
        # 1. 创建 session
        session = AgentSession(
            session_id=session_id,
            title="pytest integration test",
        )

        db.add(session)
        db.commit()
        db.refresh(session)

        assert session.id is not None
        assert session.session_id == session_id

        # 2. 写入 user message
        user_message = AgentMessage(
            session_id=session_id,
            role="user",
            content="测试用户消息",
            trace_id="pytest-trace-user",
        )

        db.add(user_message)
        db.commit()
        db.refresh(user_message)

        assert user_message.id is not None

        # 3. 写入 assistant message
        assistant_message = AgentMessage(
            session_id=session_id,
            role="assistant",
            content="测试助手回复",
            trace_id="pytest-trace-assistant",
        )

        db.add(assistant_message)
        db.commit()
        db.refresh(assistant_message)

        assert assistant_message.id is not None

        # 4. 查询 session
        saved_session = (
            db.query(AgentSession)
            .filter(
                AgentSession.session_id
                == session_id
            )
            .first()
        )

        assert saved_session is not None
        assert (
            saved_session.session_id
            == session_id
        )

        # 5. 查询 messages
        messages = (
            db.query(AgentMessage)
            .filter(
                AgentMessage.session_id
                == session_id
            )
            .order_by(
                AgentMessage.id.asc()
            )
            .all()
        )

        assert len(messages) == 2
        assert messages[0].role == "user"
        assert (
            messages[0].content
            == "测试用户消息"
        )
        assert (
            messages[1].role
            == "assistant"
        )
        assert (
            messages[1].content
            == "测试助手回复"
        )

    finally:
        # 清理测试数据
        db.query(AgentMessage).filter(
            AgentMessage.session_id
            == session_id
        ).delete()

        db.query(AgentSession).filter(
            AgentSession.session_id
            == session_id
        ).delete()

        db.commit()
        db.close()