from __future__ import annotations

from typing import Any


def normalize_history(
    history: list[dict[str, Any]] | None,
) -> list[dict[str, str]]:
    """
    清理历史消息。

    只保留：
    - role=user
    - role=assistant
    - 非空 content
    """

    if not history:
        return []

    cleaned: list[dict[str, str]] = []

    for message in history:
        role = str(
            message.get(
                "role",
                "",
            )
        ).strip()

        content = str(
            message.get(
                "content",
                "",
            )
        ).strip()

        if role not in {
            "user",
            "assistant",
        }:
            continue

        if not content:
            continue

        cleaned.append(
            {
                "role": role,
                "content": content,
            }
        )

    return cleaned


def format_history_for_prompt(
    history: list[dict[str, str]],
) -> str:
    """
    把最近对话转换成适合 LLM 阅读的文本。
    """

    history = normalize_history(
        history
    )

    if not history:
        return "暂无最近对话。"

    lines: list[str] = []

    for message in history:
        role = message["role"]

        role_name = (
            "用户"
            if role == "user"
            else "Agent"
        )

        lines.append(
            f"{role_name}："
            f"{message['content']}"
        )

    return "\n".join(
        lines
    )