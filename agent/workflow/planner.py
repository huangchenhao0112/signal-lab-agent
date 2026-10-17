from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from app.core.llm_client import LLMClient

from agent.workflow.history import (
    format_history_for_prompt,
)

from agent.workflow.state import (
    AgentState,
)


class ToolAction(BaseModel):
    """
    Planner生成的一步Tool调用。
    """

    tool_name: str

    arguments: dict[
        str,
        Any,
    ] = Field(
        default_factory=dict
    )

    purpose: str = ""


class ToolPlan(BaseModel):
    """
    完整Tool执行计划。
    """

    actions: list[
        ToolAction
    ] = Field(
        default_factory=list
    )

    reason: str = ""


def _clean_json_text(
    text: str,
) -> str:
    cleaned = text.strip()

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()

        if lines:
            lines = lines[1:]

        if (
            lines
            and lines[-1].strip()
            == "```"
        ):
            lines = lines[:-1]

        cleaned = "\n".join(
            lines
        ).strip()

    return cleaned


def _build_tools_prompt(
    tools: list[dict[str, Any]],
) -> str:
    """
    将当前MCP实际可用Tool
    转成Planner可读格式。
    """

    if not tools:
        return "当前没有可用Tool。"

    items: list[str] = []

    for tool in tools:
        items.append(
            json.dumps(
                {
                    "name": (
                        tool.get("name")
                    ),
                    "description": (
                        tool.get(
                            "description"
                        )
                    ),
                    "input_schema": (
                        tool.get(
                            "input_schema"
                        )
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    return "\n\n".join(items)


def create_planner_node(
    llm: LLMClient,
):
    """
    创建复杂任务Planner。

    Planner只负责：
    用户问题
        ↓
    Tool执行计划

    不真正执行Tool。
    """

    def planner_node(
        state: AgentState,
    ) -> dict:
        user_query = state[
            "user_query"
        ]

        available_tools = (
            state.get(
                "available_tools",
                [],
            )
        )

        summary = state.get(
            "conversation_summary",
            "",
        )

        history = state.get(
            "conversation_history",
            [],
        )

        history_text = (
            format_history_for_prompt(
                history
            )
        )

        tools_prompt = (
            _build_tools_prompt(
                available_tools
            )
        )

        system_prompt = f"""
你是 SignalLab Agent 的任务规划器。

你的任务不是回答用户问题，
而是把复杂问题拆成一组
可以顺序执行的 MCP Tool Action。

==============================
长期历史摘要
==============================

{summary or "暂无历史摘要。"}

==============================
最近对话
==============================

{history_text}

==============================
当前真实可用 MCP Tool
==============================

{tools_prompt}

==============================
规划原则
==============================

1. 只能使用上面真实存在的Tool。

2. 不得编造Tool名称。

3. arguments必须严格遵守
对应Tool的input_schema。

4. 能一次完成的任务，
不要重复调用多个Tool。

5. 复杂问题可以拆成多个Action，
但通常控制在2~5步。

6. 多模型、多指标分析优先使用：

analyze_model_performance

而不是重复调用很多次
get_model_metrics。

7. 用户要求：
“比较并解释原因”

通常需要：

结构化数据Tool
+
RAG Tool

8. 用户要求：
“比较、解释并画图”

通常需要：

结构化数据Tool
+
RAG Tool
+
generate_metric_plot

9. 复杂研究解释问题，
如果存在：

search_research_docs_multi

应优先使用它。

10. 用户只是问定义、
单一指标或单一知识问题时，
不应该进入复杂Plan。

11. 不要为了凑步骤重复查询
相同数据。

12. 用户没有要求画图时，
不要擅自调用绘图Tool。

==============================
典型例子
==============================

用户：

“比较NCSD-Net、TFA-Net、
HCTFR-Net在low_snr下的表现，
解释原因，并画peak_f1对比图。”

合理计划：

{{
    "actions": [
        {{
            "tool_name":
            "analyze_model_performance",
            "arguments": {{
                "models": [
                    "NCSD-Net",
                    "TFA-Net",
                    "HCTFR-Net"
                ],
                "group": "low_snr",
                "metrics": null,
                "exclude_metrics": null
            }},
            "purpose":
            "获得三模型真实多指标表现"
        }},
        {{
            "tool_name":
            "search_research_docs_multi",
            "arguments": {{
                "questions": [
                    "NCSD-Net低信噪比性能取舍和原因是什么？",
                    "NCSD-Net的FFT参考证据和模型结构如何影响低信噪比性能？",
                    "NCSD-Net、TFA-Net和HCTFR-Net的结构与机制差异是什么？"
                ],
                "top_k": 8,
                "per_query_k": 6
            }},
            "purpose":
            "检索性能差异的研究解释"
        }},
        {{
            "tool_name":
            "generate_metric_plot",
            "arguments": {{
                "models": [
                    "NCSD-Net",
                    "TFA-Net",
                    "HCTFR-Net"
                ],
                "metric": "peak_f1",
                "group": "low_snr"
            }},
            "purpose":
            "生成peak_f1模型对比图"
        }}
    ],
    "reason":
    "问题同时要求数据比较、机制解释和绘图"
}}

==============================
输出要求
==============================

只返回合法JSON。

格式：

{{
    "actions": [
        {{
            "tool_name": "真实Tool名称",
            "arguments": {{}},
            "purpose": "调用目的"
        }}
    ],
    "reason": "规划原因"
}}
"""

        result = llm.chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        system_prompt
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        user_query
                    ),
                },
            ],
            temperature=0.0,
        )

        if not result.ok:
            return {
                "error": (
                    "Planner LLM调用失败："
                    f"{result.error_type} - "
                    f"{result.error_message}"
                )
            }

        if not result.content:
            return {
                "error": (
                    "Planner返回为空"
                )
            }

        try:
            data = json.loads(
                _clean_json_text(
                    result.content
                )
            )

            plan = ToolPlan(
                **data
            )

        except Exception as error:
            return {
                "error": (
                    "Planner结果解析失败："
                    f"{type(error).__name__}: "
                    f"{error}"
                )
            }

        available_names = {
            str(
                tool.get("name")
            )
            for tool
            in available_tools
        }

        valid_actions: list[
            dict[str, Any]
        ] = []

        for action in plan.actions:
            if (
                action.tool_name
                not in available_names
            ):
                continue

            valid_actions.append(
                {
                    "tool_name": (
                        action.tool_name
                    ),
                    "arguments": (
                        action.arguments
                    ),
                    "purpose": (
                        action.purpose
                    ),
                }
            )

        if not valid_actions:
            return {
                "error": (
                    "Planner没有生成"
                    "可执行的Tool计划"
                )
            }

        return {
            "tool_plan": (
                valid_actions
            ),
            "plan_reason": (
                plan.reason
            ),
        }

    return planner_node