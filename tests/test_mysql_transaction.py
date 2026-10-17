from __future__ import annotations

import uuid

import pytest

from app.database.mysql import SessionLocal
from app.database.models import AgentSession


pytestmark = pytest.mark.integration


def test_mysql_rollback():
    db = SessionLocal()

    session_id = f"rollback-{uuid.uuid4()}"

    try:
        session = AgentSession(
            session_id=session_id,
            title="rollback test",
        )

        db.add(session)

        # 不 commit，直接 rollback
        db.rollback()

        saved = (
            db.query(AgentSession)
            .filter(
                AgentSession.session_id
                == session_id
            )
            .first()
        )

        assert saved is None

    finally:
        db.query(
            AgentSession
        ).filter(
            AgentSession.session_id
            == session_id
        ).delete()

        db.commit()
        db.close()