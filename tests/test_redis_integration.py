from __future__ import annotations

import asyncio
import uuid

import pytest

pytestmark = pytest.mark.integration

from agent.memory.redis_memory import (
    RedisConversationMemory,
)


REDIS_URL = (
    "redis://127.0.0.1:6379/0"
)


def test_redis_memory_crud():
    """
    Redis 集成测试：

    1. ping
    2. summary 写入 / 读取
    3. message 写入 / 读取
    4. save_turn
    5. replace_messages
    6. clear_session
    """

    asyncio.run(
        _test_redis_memory_crud()
    )


async def _test_redis_memory_crud():
    session_id = (
        f"pytest-{uuid.uuid4()}"
    )

    memory = (
        RedisConversationMemory(
            redis_url=REDIS_URL,
            ttl_seconds=300,
        )
    )

    try:
        # -------------------------
        # 1. Redis连接检查
        # -------------------------

        assert (
            await memory.ping()
            is True
        )

        # -------------------------
        # 2. 初始状态应为空
        # -------------------------

        summary = (
            await memory.load_summary(
                session_id
            )
        )

        messages = (
            await memory.load_messages(
                session_id
            )
        )

        assert summary == ""
        assert messages == []

        # -------------------------
        # 3. summary写入 / 读取
        # -------------------------

        await memory.save_summary(
            session_id=session_id,
            summary=(
                "用户正在讨论NCSD-Net。"
            ),
        )

        summary = (
            await memory.load_summary(
                session_id
            )
        )

        assert (
            summary
            == "用户正在讨论NCSD-Net。"
        )

        # -------------------------
        # 4. 单条message写入
        # -------------------------

        await memory.append_message(
            session_id=session_id,
            role="user",
            content="NCSD-Net是什么？",
        )

        messages = (
            await memory.load_messages(
                session_id
            )
        )

        assert len(messages) == 1

        assert messages[0] == {
            "role": "user",
            "content": (
                "NCSD-Net是什么？"
            ),
        }

        # -------------------------
        # 5. 保存完整一轮对话
        # -------------------------

        await memory.save_turn(
            session_id=session_id,
            user_query=(
                "它有哪些创新点？"
            ),
            assistant_answer=(
                "主要包括FFT参考证据"
                "与神经补偿机制。"
            ),
        )

        messages = (
            await memory.load_messages(
                session_id
            )
        )

        # 前面1条
        # +
        # save_turn产生2条
        assert len(messages) == 3

        assert messages[1] == {
            "role": "user",
            "content": (
                "它有哪些创新点？"
            ),
        }

        assert messages[2] == {
            "role": "assistant",
            "content": (
                "主要包括FFT参考证据"
                "与神经补偿机制。"
            ),
        }

        # -------------------------
        # 6. replace_messages
        # -------------------------

        new_messages = [
            {
                "role": "user",
                "content": (
                    "保留的最近问题"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "保留的最近回答"
                ),
            },
        ]

        await memory.replace_messages(
            session_id=session_id,
            messages=new_messages,
        )

        messages = (
            await memory.load_messages(
                session_id
            )
        )

        assert (
            messages
            == new_messages
        )

        # -------------------------
        # 7. clear_session
        # -------------------------

        await memory.clear_session(
            session_id
        )

        summary = (
            await memory.load_summary(
                session_id
            )
        )

        messages = (
            await memory.load_messages(
                session_id
            )
        )

        assert summary == ""
        assert messages == []

    finally:
        # 即使断言失败，
        # 也尽量清理测试数据
        await memory.clear_session(
            session_id
        )

        await memory.close()

def test_redis_invalid_role():
    """
    role只能是：
    user / assistant

    非法role应该抛ValueError。
    """

    asyncio.run(
        _test_redis_invalid_role()
    )


async def _test_redis_invalid_role():
    session_id = (
        f"pytest-{uuid.uuid4()}"
    )

    memory = (
        RedisConversationMemory(
            redis_url=REDIS_URL,
            ttl_seconds=300,
        )
    )

    try:
        with pytest.raises(
            ValueError,
            match="非法 role",
        ):
            await memory.append_message(
                session_id=session_id,
                role="system",
                content="非法消息",
            )

    finally:
        await memory.clear_session(
            session_id
        )

        await memory.close()

def test_redis_ttl():
    """
    保存消息后，
    Redis Key应该设置TTL。
    """

    asyncio.run(
        _test_redis_ttl()
    )


async def _test_redis_ttl():
    session_id = (
        f"pytest-{uuid.uuid4()}"
    )

    ttl_seconds = 300

    memory = (
        RedisConversationMemory(
            redis_url=REDIS_URL,
            ttl_seconds=ttl_seconds,
        )
    )

    try:
        await memory.append_message(
            session_id=session_id,
            role="user",
            content="TTL测试",
        )

        key = memory._messages_key(
            session_id
        )

        ttl = await memory.redis.ttl(
            key
        )

        assert ttl > 0

        assert (
            ttl
            <= ttl_seconds
        )

    finally:
        await memory.clear_session(
            session_id
        )

        await memory.close()