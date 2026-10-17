from __future__ import annotations

import uuid

from sqlalchemy import (
    select,
)

from app.database.models import (
    AgentSession,
)

from app.database.mysql import (
    get_db_session,
)


def main() -> None:
    session_id = str(
        uuid.uuid4()
    )

    # =========================
    # INSERT
    # =========================

    with get_db_session() as db:
        record = AgentSession(
            session_id=session_id,
            title="MySQL接入测试",
        )

        db.add(
            record
        )

    print(
        "写入成功：",
        session_id,
    )

    # =========================
    # SELECT
    # =========================

    with get_db_session() as db:
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

        record = (
            result
            .scalar_one_or_none()
        )

        if record is None:
            print(
                "没有查询到记录"
            )

            return

        print(
            "查询成功"
        )

        print(
            "ID：",
            record.id,
        )

        print(
            "Session ID：",
            record.session_id,
        )

        print(
            "Title：",
            record.title,
        )

        print(
            "Created At：",
            record.created_at,
        )


if __name__ == "__main__":
    main()