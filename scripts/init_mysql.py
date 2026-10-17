from __future__ import annotations

from app.database.mysql import (
    Base,
    check_mysql_connection,
    engine,
)

# 必须导入Model，
# SQLAlchemy才能知道有哪些表
from app.database.models import (
    AgentSession,
)


def main() -> None:
    print(
        "正在检查MySQL连接..."
    )

    if not check_mysql_connection():
        raise RuntimeError(
            "MySQL连接失败"
        )

    print(
        "MySQL连接成功"
    )

    print(
        "正在创建数据库表..."
    )

    Base.metadata.create_all(
        bind=engine
    )

    print(
        "数据库表创建完成"
    )


if __name__ == "__main__":
    main()