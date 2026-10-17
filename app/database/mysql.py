from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Generator

from dotenv import load_dotenv

from sqlalchemy import (
    create_engine,
    text,
)

from sqlalchemy.orm import (
    DeclarativeBase,
    Session,
    sessionmaker,
)


load_dotenv()


MYSQL_HOST = os.getenv(
    "MYSQL_HOST",
    "127.0.0.1",
)

MYSQL_PORT = os.getenv(
    "MYSQL_PORT",
    "3306",
)

MYSQL_DATABASE = os.getenv(
    "MYSQL_DATABASE",
    "signal_lab_agent",
)

MYSQL_USER = os.getenv(
    "MYSQL_USER",
    "signal_agent",
)

MYSQL_PASSWORD = os.getenv(
    "MYSQL_PASSWORD",
    "",
)


DATABASE_URL = (
    f"mysql+pymysql://"
    f"{MYSQL_USER}:"
    f"{MYSQL_PASSWORD}"
    f"@{MYSQL_HOST}:"
    f"{MYSQL_PORT}/"
    f"{MYSQL_DATABASE}"
    "?charset=utf8mb4"
)


class Base(
    DeclarativeBase
):
    pass


engine = create_engine(
    DATABASE_URL,

    # 取连接前检查该连接是否仍然有效
    pool_pre_ping=True,

    # 常驻连接池数量
    pool_size=5,

    # 高峰期额外允许的连接数
    max_overflow=10,

    # 获取连接最长等待时间
    pool_timeout=30,

    # 调试SQL时可以临时改为True
    echo=False,
)


SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)


def get_db() -> Generator[
    Session,
    None,
    None,
]:
    """
    FastAPI数据库依赖。

    每个HTTP请求：
        创建Session
        ↓
        注入Endpoint
        ↓
        请求结束
        ↓
        close

    commit/rollback由业务层显式控制。
    """

    db = SessionLocal()

    try:
        yield db

    finally:
        db.close()


@contextmanager
def get_db_session() -> Generator[
    Session,
    None,
    None,
]:
    """
    非FastAPI场景使用。

    例如：
        scripts/
        CLI
        初始化脚本
    """

    db = SessionLocal()

    try:
        yield db
        db.commit()

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()


def check_mysql_connection() -> bool:
    """
    MySQL健康检查。
    """

    try:
        with engine.connect() as connection:
            connection.execute(
                text("SELECT 1")
            )

        return True

    except Exception:
        return False