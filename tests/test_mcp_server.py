import json

import anyio
from mcp import Client
from mcp.types import TextContent

from mcp_server.server import mcp


def print_tool_result(
    title: str,
    result,
) -> None:
    """打印MCP工具调用结果。"""
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    print(
        f"is_error：{result.is_error}"
    )

    if result.structured_content is not None:
        print("structured_content：")

        print(
            json.dumps(
                result.structured_content,
                ensure_ascii=False,
                indent=2,
            )
        )

    print("content：")

    for block in result.content:
        if isinstance(block, TextContent):
            print(block.text)
        else:
            print(block)


async def main() -> None:
    """
    使用进程内MCP Client测试Server。

    不启动HTTP端口，
    不调用LLM，
    不需要FastAPI或Swagger。
    """
    async with Client(
        mcp,
        raise_exceptions=True,
    ) as client:
        print("=" * 70)
        print("MCP Server连接成功")
        print("=" * 70)

        print(
            f"协议版本：{client.protocol_version}"
        )

        if client.server_info is not None:
            print(
                f"Server名称："
                f"{client.server_info.name}"
            )

        tools_result = await client.list_tools()

        tool_names = [
            tool.name
            for tool in tools_result.tools
        ]

        print(
            f"发现的工具：{tool_names}"
        )

        expected_tools = {
            "get_model_metrics",
            "compare_models",
        }

        if set(tool_names) != expected_tools:
            raise RuntimeError(
                "MCP工具列表不符合预期："
                f"{tool_names}"
            )

        metrics_result = await client.call_tool(
            "get_model_metrics",
            {
                "model": "NCSD-Net",
                "metric": "hit_rate",
                "group": "overall",
            },
        )

        print_tool_result(
            "测试1：查询NCSD-Net的hit_rate",
            metrics_result,
        )

        if metrics_result.is_error:
            raise RuntimeError(
                "get_model_metrics调用失败"
            )

        compare_result = await client.call_tool(
            "compare_models",
            {
                "model_a": "NCSD-Net",
                "model_b": "TFA-Net",
                "metric": "hit_rate",
                "group": "overall",
            },
        )

        print_tool_result(
            "测试2：比较NCSD-Net和TFA-Net",
            compare_result,
        )

        if compare_result.is_error:
            raise RuntimeError(
                "compare_models调用失败"
            )

        invalid_result = await client.call_tool(
            "get_model_metrics",
            {
                "model": "不存在的模型",
                "metric": "hit_rate",
                "group": "overall",
            },
        )

        print_tool_result(
            "测试3：不存在模型的错误返回",
            invalid_result,
        )

        if not invalid_result.is_error:
            raise RuntimeError(
                "无效模型应该返回工具错误"
            )

        print()
        print("=" * 70)
        print("MCP Server三项测试全部完成")
        print("=" * 70)


if __name__ == "__main__":
    anyio.run(main)