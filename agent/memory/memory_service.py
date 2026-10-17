from __future__ import annotations

from app.core.llm_client import LLMClient

from agent.memory.redis_memory import (
    RedisConversationMemory,
)


class ConversationMemoryService:
    """
    SignalLab Agent 会话记忆服务。

    负责：
    - 加载长期摘要
    - 加载最近对话
    - 保存一轮对话
    - 压缩较老对话为摘要
    """

    def __init__(
        self,
        memory: RedisConversationMemory,
        llm: LLMClient,
        summary_trigger_messages: int = 14,
        recent_window_messages: int = 8,
    ):
        self.memory = memory
        self.llm = llm

        self.summary_trigger_messages = (
            summary_trigger_messages
        )

        self.recent_window_messages = (
            recent_window_messages
        )

    async def load_context(
        self,
        session_id: str,
    ) -> tuple[
        str,
        list[dict[str, str]],
    ]:
        """
        返回：

        summary
        +
        最近原始对话窗口
        """

        summary = (
            await self.memory.load_summary(
                session_id
            )
        )

        messages = (
            await self.memory.load_messages(
                session_id
            )
        )

        # 这里只给 Agent 最近窗口，
        # Redis 本身可能暂时保存更多，
        # 用于后面做摘要。
        recent_messages = (
            messages[
                -self.recent_window_messages:
            ]
        )

        return (
            summary,
            recent_messages,
        )

    async def save_turn(
        self,
        session_id: str,
        user_query: str,
        assistant_answer: str,
    ) -> None:
        """
        保存一整轮对话。
        """

        await self.memory.save_turn(
            session_id=session_id,
            user_query=user_query,
            assistant_answer=assistant_answer,
        )

    async def compact_if_needed(
        self,
        session_id: str,
    ) -> None:
        """
        当原始消息达到阈值后：

        旧消息
        → LLM 压缩为摘要

        最近几条
        → 保留原文
        """

        messages = (
            await self.memory.load_messages(
                session_id
            )
        )

        if len(messages) < (
            self.summary_trigger_messages
        ):
            return

        keep_count = (
            self.recent_window_messages
        )

        old_messages = messages[
            :-keep_count
        ]

        recent_messages = messages[
            -keep_count:
        ]

        if not old_messages:
            return

        old_summary = (
            await self.memory.load_summary(
                session_id
            )
        )

        old_history_text = "\n".join(
            (
                (
                    "用户："
                    if item["role"] == "user"
                    else "Agent："
                )
                + item["content"]
            )
            for item in old_messages
        )

        result = self.llm.chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你负责维护 SignalLab Agent "
                        "的长期会话记忆。"
                        "请把较早的对话压缩成简洁摘要。"
                        "必须保留后续对话可能需要的信息，"
                        "例如模型名称、实验指标、数据分组、"
                        "用户目标、已经得到的结论、"
                        "用户明确提出的要求、"
                        "以及尚未解决的问题。"
                        "不要记录无关寒暄。"
                        "不要编造不存在的信息。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "已有历史摘要：\n"
                        f"{old_summary or '无'}"
                        "\n\n"
                        "需要压缩的较早对话：\n"
                        f"{old_history_text}"
                        "\n\n"
                        "请输出更新后的完整摘要。"
                    ),
                },
            ],
            temperature=0.0,
        )

        if not result.ok:
            return

        if not result.content:
            return

        # 保存新摘要
        await self.memory.save_summary(
            session_id=session_id,
            summary=result.content,
        )

        # 删除已经进入摘要的旧消息，
        # 只保留最近窗口
        await self.memory.replace_messages(
            session_id=session_id,
            messages=recent_messages,
        )

    async def clear(
        self,
        session_id: str,
    ) -> None:
        """
        清除一个 session。
        """

        await self.memory.clear_session(
            session_id
        )