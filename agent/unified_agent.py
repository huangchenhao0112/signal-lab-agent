from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path

from mcp import (
    Client,
    StdioServerParameters,
)

from app.core.config import Settings
from app.core.llm_client import LLMClient

from agent.memory.redis_memory import (
    RedisConversationMemory,
)
from agent.memory.memory_service import (
    ConversationMemoryService,
)
from agent.workflow.graph import (
    build_agent_graph,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

SERVER_FILE = (
    PROJECT_ROOT
    / "mcp_server"
    / "server.py"
)


async def main() -> None:
    """
    SignalLab Unified Agent
    最终入口。
    """

    # =========================
    # 1. Settings + LLM
    # =========================

    settings = Settings()

    llm = LLMClient(
        settings=settings
    )

    # =========================
    # 2. Redis Memory
    # =========================

    memory_store = (
        RedisConversationMemory(
            redis_url=(
                settings.redis_url
            ),
        )
    )

    memory_service = (
        ConversationMemoryService(
            memory=memory_store,
            llm=llm,
            summary_trigger_messages=14,
            recent_window_messages=8,
        )
    )

    try:
        await memory_store.ping()

        print(
            "Redis Memory 连接成功。"
        )

    except Exception as error:
        print(
            "Redis Memory 连接失败："
            f"{type(error).__name__}: "
            f"{error}"
        )

        await memory_store.close()

        llm.close()

        return

    # =========================
    # 3. Session
    # =========================

    session_id = input(
        "请输入 session_id "
        "（直接回车创建新会话）："
    ).strip()

    if not session_id:
        session_id = str(
            uuid.uuid4()
        )

        print(
            "已创建新会话。"
        )

    else:
        print(
            "正在恢复已有会话。"
        )

    print(
        f"当前 session_id："
        f"{session_id}"
    )

    # =========================
    # 4. MCP
    # =========================

    server_params = (
        StdioServerParameters(
            command=sys.executable,
            args=[
                str(SERVER_FILE)
            ],
        )
    )

    print(
        "正在启动 SignalLab "
        "Unified Agent..."
    )

    try:
        async with Client(
            server_params
        ) as client:

            tools_result = (
                await client.list_tools()
            )

            tools = (
                tools_result.tools
            )

            available_tools = [
                {
                    "name": (
                        tool.name
                    ),
                    "description": (
                        tool.description
                    ),
                    "input_schema": (
                        tool.input_schema
                    ),
                }
                for tool in tools
            ]

            print(
                "MCP Server 连接成功。"
            )

            print(
                f"发现 {len(tools)} "
                "个 MCP Tool："
            )

            for tool in tools:
                print(
                    f"- {tool.name}"
                )

            # =====================
            # 5. LangGraph
            # =====================

            graph = build_agent_graph(
                llm=llm,
                client=client,
                tools=tools,
            )

            print(
                "\nLangGraph Unified Agent "
                "已启动。"
            )

            print(
                "直接输入自然语言问题即可。"
            )

            print(
                "输入 exit / quit / q 退出。"
            )

            # =====================
            # 6. Main Loop
            # =====================

            while True:
                query = input(
                    "\n你："
                ).strip()

                if not query:
                    continue

                if query.lower() in {
                    "exit",
                    "quit",
                    "q",
                }:
                    print(
                        "已退出 Unified Agent。"
                    )
                    break

                # -----------------
                # 读取 Redis Memory
                # -----------------

                summary, history = (
                    await memory_service
                    .load_context(
                        session_id
                    )
                )

                try:
                    result = (
                        await graph.ainvoke(
                            {
                                "user_query": (
                                    query
                                ),
                                "session_id": (
                                    session_id
                                ),
                                "conversation_summary": (
                                    summary
                                ),
                                "conversation_history": (
                                    history
                                ),
                                "available_tools": (
                                    available_tools
                                ),
                                "top_k": 3,
                            }
                        )
                    )

                    print(
                        "tool_plan:",
                        result.get(
                            "tool_plan"
                        )
                    )

                    print(
                        "tools_used:",
                        result.get(
                            "tools_used"
                        )
                    )

                    print(
                        "tool_results:",
                        result.get(
                            "tool_results"
                        )
                    )

                    print(
                        "最终metric_arguments:",
                        result.get(
                            "metric_arguments"
                        )
                    )
                    
                except Exception as error:
                    print(
                        "\nAgent 执行异常："
                    )

                    print(
                        f"{type(error).__name__}: "
                        f"{error}"
                    )

                    continue

                answer = result.get(
                    "final_answer",
                    "没有生成最终回答。",
                )

                print(
                    "\nAgent："
                )

                print(
                    answer
                )

                # -----------------
                # 保存当前一轮
                # -----------------

                await (
                    memory_service
                    .save_turn(
                        session_id=(
                            session_id
                        ),
                        user_query=query,
                        assistant_answer=(
                            answer
                        ),
                    )
                )

                # -----------------
                # 必要时压缩历史
                # -----------------

                await (
                    memory_service
                    .compact_if_needed(
                        session_id
                    )
                )

    finally:
        await memory_store.close()

        llm.close()


if __name__ == "__main__":
    asyncio.run(
        main()
    )