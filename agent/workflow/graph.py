from __future__ import annotations

from typing import Any

from langgraph.graph import (
    END,
    START,
    StateGraph,
)

from mcp import Client

from app.core.llm_client import LLMClient

from agent.workflow.state import (
    AgentState,
)
from agent.workflow.router import (
    create_router_node,
)
from agent.workflow.planner import (
    create_planner_node,
)
from agent.workflow.nodes import (
    capability_node,
    create_answer_node,
    create_metric_node,
    create_rag_node,
    create_tool_executor_node,
    fallback_node,
)


def route_after_router(
    state: AgentState,
) -> str:
    """
    Router Node 后的条件路由。
    """

    if state.get("error"):
        return "fallback"

    route = state.get(
        "route",
        "none",
    )

    if route == "metric":
        return "metric"

    if route == "rag":
        return "rag"

    if route == "hybrid":
        return "hybrid"

    if route == "plan":
        return "planner"

    if route == "capability":
        return "capability"
    
    return "fallback"


def route_after_metric(
    state: AgentState,
) -> str:
    """
    metric_node 执行完成后：

    - hybrid 继续去 RAG
    - metric 直接生成答案
    """

    if state.get("error"):
        return "answer"

    if state.get("route") == "hybrid":
        return "rag"

    return "answer"


def build_agent_graph(
    llm: LLMClient,
    client: Client,
    tools: list[Any],
):
    """
    构建 SignalLab Unified Agent LangGraph。
    """

    builder = StateGraph(
        AgentState
    )

    # -------------------------
    # Node
    # -------------------------

    builder.add_node(
        "router",
        create_router_node(
            llm=llm,
            tools=tools,
        ),
    )

    builder.add_node(
        "metric",
        create_metric_node(
            client=client,
            llm=llm,
        ),
    )

    builder.add_node(
        "rag",
        create_rag_node(
            client=client,
        ),
    )

    builder.add_node(
        "answer",
        create_answer_node(
            llm=llm,
        ),
    )

    builder.add_node(
        "capability",
        capability_node,
    )

    builder.add_node(
        "planner",
        create_planner_node(
            llm=llm,
        ),
    )

    builder.add_node(
        "tool_executor",
        create_tool_executor_node(
            client=client,
            llm=llm,
        ),
    )

    builder.add_node(
        "fallback",
        fallback_node,
    )

    # -------------------------
    # START → Router
    # -------------------------

    builder.add_edge(
        START,
        "router",
    )

    # -------------------------
    # Router 条件分支
    # -------------------------

    builder.add_conditional_edges(
        "router",
        route_after_router,
        {
            "metric": "metric",
            "rag": "rag",
            "hybrid": "metric",
            "fallback": "fallback",
            "capability": "capability",
            "planner": "planner",
        },
    )

    # -------------------------
    # Metric 后续
    # -------------------------

    builder.add_conditional_edges(
        "metric",
        route_after_metric,
        {
            "rag": "rag",
            "answer": "answer",
        },
    )

    # -------------------------
    # RAG → Answer
    # -------------------------

    builder.add_edge(
        "rag",
        "answer",
    )

    # -------------------------
    # Answer / Fallback → END
    # -------------------------

    builder.add_edge(
        "answer",
        END,
    )

    builder.add_edge(
        "fallback",
        END,
    )
    builder.add_edge(
        "capability",
        END,
    )
    builder.add_edge(
        "planner",
        "tool_executor",
    )
    builder.add_edge(
        "tool_executor",
        "answer",
    )
    return builder.compile()