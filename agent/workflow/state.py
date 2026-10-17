from __future__ import annotations

from typing import Any, TypedDict
import operator

from typing import (
    Annotated,
    Any,
    TypedDict,
)

class AgentState(TypedDict, total=False):
    """
    SignalLab Unified Agent 的共享状态。

    LangGraph 中的各个 Node
    都通过这个 State 交换数据。
    """

    # 用户输入
    user_query: str
    
    # 当前会话 ID
    session_id: str

    # =========================
    # Memory
    # =========================

    # Redis 中保存的历史摘要
    conversation_summary: str

    conversation_history: list[
        dict[str, str]
    ]

    # 当前 MCP Server 实际拥有的 Tool
    available_tools: list[dict[str, Any]]

    # 路由结果
    route: str
    route_reason: str

    # MCP 实验指标调用
    metric_tool: str
    metric_arguments: dict[str, Any]

    # RAG 检索
    rag_question: str
    rag_queries: list[str]
    top_k: int

    # =========================
    # Planner
    # =========================

    tool_plan: list[
        dict[str, Any]
    ]

    tool_results: list[
        dict[str, Any]
    ]

    tools_used: Annotated[
        list[str],
        operator.add,
    ]

    plan_reason: str

    # Tool 返回结果
    metric_result: str
    rag_result: str

    # 最终结果
    final_answer: str

    # 错误信息
    error: str