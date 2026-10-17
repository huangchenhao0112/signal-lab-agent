from __future__ import annotations

import json

import redis.asyncio as redis


class RedisConversationMemory:
    """
    Redis 会话记忆存储层。

    每个 session 保存：

    1. summary
       历史长期摘要

    2. messages
       最近和待压缩的原始对话
    """

    def __init__(
        self,
        redis_url: str,
        ttl_seconds: int = 60 * 60 * 24 * 30,
    ):
        self.redis = redis.from_url(
            redis_url,
            encoding="utf-8",
            decode_responses=True,
        )

        self.ttl_seconds = ttl_seconds

    def _summary_key(
        self,
        session_id: str,
    ) -> str:
        return (
            "signal_lab:"
            f"session:{session_id}:summary"
        )

    def _messages_key(
        self,
        session_id: str,
    ) -> str:
        return (
            "signal_lab:"
            f"session:{session_id}:messages"
        )

    async def ping(
        self,
    ) -> bool:
        """
        检查 Redis 服务是否可用。
        """

        result = await self.redis.ping()

        return bool(result)

    async def load_summary(
        self,
        session_id: str,
    ) -> str:
        """
        读取历史摘要。
        """

        value = await self.redis.get(
            self._summary_key(
                session_id
            )
        )

        return value or ""

    async def save_summary(
        self,
        session_id: str,
        summary: str,
    ) -> None:
        """
        保存历史摘要。
        """

        key = self._summary_key(
            session_id
        )

        await self.redis.set(
            key,
            summary,
        )

        await self.redis.expire(
            key,
            self.ttl_seconds,
        )

    async def load_messages(
        self,
        session_id: str,
    ) -> list[dict[str, str]]:
        """
        读取当前 session 的原始消息。
        """

        key = self._messages_key(
            session_id
        )

        values = await self.redis.lrange(
            key,
            0,
            -1,
        )

        messages: list[
            dict[str, str]
        ] = []

        for value in values:
            try:
                item = json.loads(
                    value
                )

            except json.JSONDecodeError:
                continue

            role = item.get(
                "role"
            )

            content = item.get(
                "content"
            )

            if role not in {
                "user",
                "assistant",
            }:
                continue

            if not isinstance(
                content,
                str,
            ):
                continue

            if not content.strip():
                continue

            messages.append(
                {
                    "role": role,
                    "content": content,
                }
            )

        return messages

    async def append_message(
        self,
        session_id: str,
        role: str,
        content: str,
    ) -> None:
        """
        追加一条消息。
        """

        if role not in {
            "user",
            "assistant",
        }:
            raise ValueError(
                f"非法 role：{role}"
            )

        key = self._messages_key(
            session_id
        )

        value = json.dumps(
            {
                "role": role,
                "content": content,
            },
            ensure_ascii=False,
        )

        await self.redis.rpush(
            key,
            value,
        )

        await self.redis.expire(
            key,
            self.ttl_seconds,
        )

    async def save_turn(
        self,
        session_id: str,
        user_query: str,
        assistant_answer: str,
    ) -> None:
        """
        保存完整一轮对话。
        """

        await self.append_message(
            session_id=session_id,
            role="user",
            content=user_query,
        )

        await self.append_message(
            session_id=session_id,
            role="assistant",
            content=assistant_answer,
        )

    async def replace_messages(
        self,
        session_id: str,
        messages: list[dict[str, str]],
    ) -> None:
        """
        用新的短期窗口替换原始消息。

        用于历史摘要完成后：
        删除已经被压缩进 summary 的旧消息，
        只保留最近几轮。
        """

        key = self._messages_key(
            session_id
        )

        await self.redis.delete(
            key
        )

        if not messages:
            return

        values = [
            json.dumps(
                message,
                ensure_ascii=False,
            )
            for message in messages
        ]

        await self.redis.rpush(
            key,
            *values,
        )

        await self.redis.expire(
            key,
            self.ttl_seconds,
        )

    async def clear_session(
        self,
        session_id: str,
    ) -> None:
        """
        删除指定 session 的全部记忆。
        """

        await self.redis.delete(
            self._summary_key(
                session_id
            ),
            self._messages_key(
                session_id
            ),
        )

    async def close(
        self,
    ) -> None:
        """
        关闭 Redis 连接。
        """

        await self.redis.aclose()