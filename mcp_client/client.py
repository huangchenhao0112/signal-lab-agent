from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from mcp import Client, StdioServerParameters


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SERVER_FILE = PROJECT_ROOT / "mcp_server" / "server.py"


def _parse_result_content(
    content: list[Any],
) -> Any:
    """
    尝试把 MCP Tool 返回的文本内容解析成 JSON。

    如果无法解析，则返回原始文本。
    """

    if not content:
        return None

    texts: list[str] = []

    for item in content:
        text = getattr(
            item,
            "text",
            None,
        )

        if text is not None:
            texts.append(text)

    if not texts:
        return content

    combined_text = "\n".join(texts)

    try:
        return json.loads(
            combined_text
        )
    except json.JSONDecodeError:
        return combined_text


def _print_result(
    result: Any,
) -> None:
    """
    统一打印 MCP Tool 调用结果。
    """

    if result.is_error:
        print("\n调用失败：")

        parsed = _parse_result_content(
            result.content
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

    parsed = _parse_result_content(
        result.content
    )

    print("\n查询成功：")

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


async def query_model_metrics(
    client: Client,
) -> None:
    """
    交互式查询单个模型指标。
    """

    print("\n===== 查询单个模型指标 =====")

    model = input(
        "模型名称："
    ).strip()

    metric = input(
        "指标名称："
    ).strip()

    group = input(
        "数据分组："
    ).strip()

    if not model:
        print("模型名称不能为空。")
        return

    if not metric:
        print("指标名称不能为空。")
        return

    if not group:
        print("数据分组不能为空。")
        return

    result = await client.call_tool(
        "get_model_metrics",
        {
            "model": model,
            "metric": metric,
            "group": group,
        },
    )

    _print_result(result)


async def compare_model_metrics(
    client: Client,
) -> None:
    """
    交互式比较两个模型。
    """

    print("\n===== 比较两个模型 =====")

    model_a = input(
        "模型 A："
    ).strip()

    model_b = input(
        "模型 B："
    ).strip()

    metric = input(
        "指标名称："
    ).strip()

    group = input(
        "数据分组："
    ).strip()

    if not model_a:
        print("模型 A 不能为空。")
        return

    if not model_b:
        print("模型 B 不能为空。")
        return

    if not metric:
        print("指标名称不能为空。")
        return

    if not group:
        print("数据分组不能为空。")
        return

    result = await client.call_tool(
        "compare_models",
        {
            "model_a": model_a,
            "model_b": model_b,
            "metric": metric,
            "group": group,
        },
    )

    _print_result(result)


async def show_tools(
    client: Client,
) -> None:
    """
    查看 MCP Server 暴露的 Tool。
    """

    result = await client.list_tools()

    print("\n===== MCP Tools =====")

    for tool in result.tools:
        print(
            f"\n工具名称：{tool.name}"
        )

        print(
            f"说明：{tool.description}"
        )

        print(
            "参数："
        )

        properties = (
            tool.input_schema
            .get(
                "properties",
                {},
            )
        )

        required = (
            tool.input_schema
            .get(
                "required",
                [],
            )
        )

        for name, schema in (
            properties.items()
        ):
            data_type = schema.get(
                "type",
                "unknown",
            )

            required_text = (
                "必填"
                if name in required
                else "可选"
            )

            print(
                f"  - {name}"
                f" ({data_type}, "
                f"{required_text})"
            )


async def main() -> None:
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

    async with Client(
        server_params
    ) as client:

        print(
            "MCP Server 连接成功。"
        )

        while True:
            print(
                "\n"
                "==============================\n"
                "SignalLab MCP Client\n"
                "==============================\n"
                "1. 查询单个模型指标\n"
                "2. 比较两个模型\n"
                "3. 查看 MCP Tools\n"
                "0. 退出\n"
            )

            choice = input(
                "请选择功能："
            ).strip()

            if choice == "1":
                await query_model_metrics(
                    client
                )

            elif choice == "2":
                await compare_model_metrics(
                    client
                )

            elif choice == "3":
                await show_tools(
                    client
                )

            elif choice == "0":
                print(
                    "已退出 MCP Client。"
                )
                break

            else:
                print(
                    "无效选项，"
                    "请输入 0、1、2 或 3。"
                )


if __name__ == "__main__":
    asyncio.run(main())