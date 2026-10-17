from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from mcp import Client, StdioServerParameters

from app.core.config import Settings
from app.core.llm_client import LLMClient


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SERVER_FILE = PROJECT_ROOT / "mcp_server" / "server.py"


class ToolDecision(BaseModel):
    """
    LLM 对 MCP Tool 的调用决策。
    """

    tool_name: Literal[
        "get_model_metrics",
        "compare_models",
        "none",
    ]

    arguments: dict[str, Any] = Field(
        default_factory=dict
    )

    reason: str = ""


def build_tools_prompt(
    tools: list[Any],
) -> str:
    """
    把 MCP Server 暴露出来的 Tool 信息
    转换成 LLM 可以阅读的文本。

    不在 Client 中手写 Tool 参数，
    直接使用 MCP Server 返回的 input_schema。
    """

    tool_infos: list[str] = []

    for tool in tools:
        tool_info = {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema,
        }

        tool_infos.append(
            json.dumps(
                tool_info,
                ensure_ascii=False,
                indent=2,
            )
        )

    return "\n\n".join(tool_infos)


def clean_json_text(
    text: str,
) -> str:
    """
    清理 LLM 可能返回的 Markdown JSON 代码块。
    """

    cleaned = text.strip()

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()

        if lines:
            lines = lines[1:]

        if (
            lines
            and lines[-1].strip() == "```"
        ):
            lines = lines[:-1]

        cleaned = "\n".join(lines).strip()

    return cleaned


def parse_tool_decision(
    text: str,
) -> ToolDecision:
    """
    把 LLM 输出解析成 ToolDecision。
    """

    cleaned = clean_json_text(text)

    data = json.loads(cleaned)

    return ToolDecision(**data)


def decide_tool(
    llm: LLMClient,
    user_query: str,
    tools: list[Any],
) -> ToolDecision:
    """
    调用 LLM，根据自然语言决定：
    1. 调用哪个 Tool；
    2. Tool 使用哪些参数。
    """

    tools_prompt = build_tools_prompt(
        tools
    )

    system_prompt = f"""
你是 SignalLab Agent 的 MCP Tool 路由器。

你的任务不是回答实验问题本身，
而是根据用户问题决定是否调用 MCP Tool。

当前 MCP Server 提供以下工具：

{tools_prompt}

你必须遵守以下规则：

1. 如果用户查询单个模型的实验指标，
使用 get_model_metrics。

2. 如果用户要求比较两个模型的同一个指标，
使用 compare_models。

3. arguments 中的参数名称必须严格遵守
对应 Tool 的 input_schema。

4. 不允许自行增加不存在的参数。

5. 模型名称、指标名称、数据分组应尽量
保留用户原始表达。

6. 不允许凭空编造实验结果。

7. 如果现有 Tool 无法处理用户问题，
tool_name 必须返回 none。

8. 只允许返回合法 JSON，
不要使用 Markdown，不要使用 ```json。

返回格式必须严格如下：

{{
    "tool_name": "工具名或none",
    "arguments": {{
    }},
    "reason": "选择这个工具的简短原因"
}}
"""

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_query,
        },
    ]

    llm_result = llm.chat(
        messages=messages,
        temperature=0.0,
    )

    if not llm_result.ok:
        raise RuntimeError(
            "LLM调用失败："
            f"{llm_result.error_type} - "
            f"{llm_result.error_message}"
        )

    if not llm_result.content:
        raise ValueError(
            "LLM返回内容为空"
        )

    return parse_tool_decision(
        llm_result.content
    )


async def execute_tool(
    client: Client,
    decision: ToolDecision,
) -> Any:
    """
    根据 LLM 的 ToolDecision
    调用真正的 MCP Tool。
    """

    if decision.tool_name == "none":
        return None

    result = await client.call_tool(
        decision.tool_name,
        decision.arguments,
    )

    return result


def extract_mcp_text(
    result: Any,
) -> str:
    """
    从 MCP CallToolResult 中提取文本。
    """

    if not result.content:
        return ""

    texts: list[str] = []

    for item in result.content:
        text = getattr(
            item,
            "text",
            None,
        )

        if text:
            texts.append(text)

    return "\n".join(texts)


def parse_mcp_text(
    text: str,
) -> Any:
    """
    MCP Tool 当前返回 JSON 文本时，
    尝试解析成 Python 对象。
    """

    if not text:
        return None

    try:
        return json.loads(text)

    except json.JSONDecodeError:
        return text


def print_tool_decision(
    decision: ToolDecision,
) -> None:
    """
    调试阶段打印 LLM 的 Tool 决策。
    """

    print(
        "\n===== Tool Decision ====="
    )

    print(
        json.dumps(
            decision.model_dump(),
            ensure_ascii=False,
            indent=2,
        )
    )


def print_tool_result(
    result: Any,
) -> None:
    """
    显示 MCP Tool 返回结果。
    """

    text = extract_mcp_text(
        result
    )

    parsed = parse_mcp_text(
        text
    )

    if result.is_error:
        print(
            "\n===== Tool Error ====="
        )

        if isinstance(parsed, dict):
            error_message = parsed.get(
                "error_message"
            )

            if error_message:
                print(error_message)
            else:
                print(
                    json.dumps(
                        parsed,
                        ensure_ascii=False,
                        indent=2,
                    )
                )
        else:
            print(parsed)

        return

    print(
        "\n===== Tool Result ====="
    )

    if isinstance(parsed, dict):
        print(
            json.dumps(
                parsed,
                ensure_ascii=False,
                indent=2,
            )
        )

    else:
        print(parsed)


def build_final_answer(
    llm: LLMClient,
    user_query: str,
    tool_result_text: str,
) -> str:
    """
    把 Tool 的结构化结果整理成
    更自然的中文回答。

    Tool 负责提供真实数据；
    LLM 只负责组织表达。
    """

    messages = [
        {
            "role": "system",
            "content": (
                "你是 SignalLab Agent。"
                "请根据 Tool 返回的真实实验数据"
                "回答用户问题。"
                "不能修改、杜撰或猜测实验数值。"
                "回答简洁、准确。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"用户问题：\n"
                f"{user_query}\n\n"
                f"Tool真实结果：\n"
                f"{tool_result_text}"
            ),
        },
    ]

    result = llm.chat(
        messages=messages,
        temperature=0.2,
    )

    if not result.ok:
        return (
            "Tool 已成功执行，但生成自然语言回答失败："
            f"{result.error_message}"
        )

    return (
        result.content
        or "未生成最终回答。"
    )


async def main() -> None:
    """
    SignalLab MCP Agent 主程序。
    """

    # 1. 初始化项目配置
    settings = Settings()

    # 2. 初始化已有 LLMClient
    llm = LLMClient(
        settings=settings
    )

    # 3. 配置 stdio MCP Server
    server_params = (
        StdioServerParameters(
            command=sys.executable,
            args=[
                str(SERVER_FILE)
            ],
        )
    )

    print(
        "正在连接 MCP Server..."
    )

    try:
        async with Client(
            server_params
        ) as client:

            print(
                "MCP Server 连接成功。"
            )

            # 4. 动态发现 MCP Tool
            tools_result = (
                await client.list_tools()
            )

            tools = tools_result.tools

            print(
                f"已发现 {len(tools)} 个 MCP Tool："
            )

            for tool in tools:
                print(
                    f"- {tool.name}"
                )

            print(
                "\nSignalLab MCP Agent 已启动。"
            )

            print(
                "直接输入自然语言问题即可。"
            )

            print(
                "输入 exit / quit / q 退出。"
            )

            while True:
                user_query = input(
                    "\n你："
                ).strip()

                if not user_query:
                    continue

                if user_query.lower() in {
                    "exit",
                    "quit",
                    "q",
                }:
                    print(
                        "已退出 SignalLab MCP Agent。"
                    )
                    break

                # 5. LLM 决定 Tool
                try:
                    decision = decide_tool(
                        llm=llm,
                        user_query=user_query,
                        tools=tools,
                    )

                except json.JSONDecodeError as error:
                    print(
                        "\nLLM返回的Tool决策"
                        "不是合法JSON。"
                    )

                    print(
                        f"错误：{error}"
                    )

                    continue

                except Exception as error:
                    print(
                        "\nTool决策失败："
                    )

                    print(
                        f"{type(error).__name__}: "
                        f"{error}"
                    )

                    continue

                print_tool_decision(
                    decision
                )

                # 6. 当前 MCP Tool 无法处理
                if (
                    decision.tool_name
                    == "none"
                ):
                    print(
                        "\nAgent："
                        "当前 MCP Tool "
                        "无法处理这个问题。"
                    )

                    continue

                # 7. 调用 MCP Tool
                try:
                    tool_result = (
                        await execute_tool(
                            client=client,
                            decision=decision,
                        )
                    )

                except Exception as error:
                    print(
                        "\nMCP Tool 调用异常："
                    )

                    print(
                        f"{type(error).__name__}: "
                        f"{error}"
                    )

                    continue

                # 8. Tool 执行失败
                if tool_result.is_error:
                    print_tool_result(
                        tool_result
                    )

                    continue

                # 9. 显示 Tool 原始结果
                print_tool_result(
                    tool_result
                )

                # 10. LLM 根据真实 Tool 数据
                #     生成自然语言回答
                tool_text = (
                    extract_mcp_text(
                        tool_result
                    )
                )

                final_answer = (
                    build_final_answer(
                        llm=llm,
                        user_query=user_query,
                        tool_result_text=tool_text,
                    )
                )

                print(
                    "\n===== Agent Answer ====="
                )

                print(
                    final_answer
                )

    finally:
        # 11. 关闭 httpx Client
        llm.close()


if __name__ == "__main__":
    asyncio.run(main())