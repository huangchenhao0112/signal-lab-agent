from __future__ import annotations

import json
from typing import Any

from mcp import Client

from app.core.llm_client import LLMClient

from agent.workflow.history import (
    format_history_for_prompt,
)
from agent.workflow.state import (
    AgentState,
)


def _extract_mcp_text(
    result: Any,
) -> str:
    """
    提取 MCP Tool 返回文本。
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
            texts.append(
                text
            )

    return "\n".join(
        texts
    )

def _get_tool_schema(
    state: AgentState,
    tool_name: str,
) -> dict[str, Any]:
    """
    从当前 MCP Tool 列表中找到
    指定 Tool 的 input_schema。
    """

    tools = state.get(
        "available_tools",
        [],
    )

    for tool in tools:
        if (
            tool.get("name")
            == tool_name
        ):
            schema = tool.get(
                "input_schema",
                {},
            )

            if isinstance(
                schema,
                dict,
            ):
                return schema

    return {}


def _clean_llm_json(
    text: str,
) -> str:
    """
    清理 LLM 可能返回的
    ```json ... ```
    """

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


def _repair_tool_arguments(
    llm: LLMClient,
    state: AgentState,
    tool_name: str,
    arguments: dict[str, Any],
    error_message: str,
) -> tuple[
    dict[str, Any] | None,
    str,
]:
    """
    根据：

    - 用户原问题
    - Tool Schema
    - 原始参数
    - Tool真实错误

    修正一次 Tool 参数。

    不负责再次调用 Tool。
    """

    schema = _get_tool_schema(
        state,
        tool_name,
    )

    user_query = state.get(
        "user_query",
        "",
    )

    system_prompt = """
你是 SignalLab Agent 的 Tool 参数修复器。

当前某个 MCP Tool 调用失败。

你的任务不是回答用户问题，
而是根据：

1. 用户原始问题
2. Tool input_schema
3. 原始 arguments
4. Tool 返回的真实错误信息

修正 arguments。

规则：

1. Tool 名称绝对不能修改。

2. 只能修改 arguments。

3. 必须严格遵守 input_schema。

4. 优先根据 Tool 错误中提供的
“可选值”“合法值”等信息修复。

5. 可以修正明显的命名差异，例如：

PSNR -> psnr_db

Renyi -> renyi_entropy

low snr -> low_snr

但只有在错误信息、
Schema或已有上下文能够支持时才修改。

6. 不得编造不存在的模型、
指标、group或实验数据。

7. 如果无法可靠修复，
返回 can_retry=false。

8. 最多只用于重试一次，
因此不要进行试探性猜测。

只返回合法 JSON。

格式：

{
    "can_retry": true,
    "arguments": {},
    "reason": "为什么这样修正"
}
"""

    user_prompt = (
        "用户原始问题：\n"
        f"{user_query}"
        "\n\n"
        "Tool名称：\n"
        f"{tool_name}"
        "\n\n"
        "Tool input_schema：\n"
        f"{json.dumps(schema, ensure_ascii=False)}"
        "\n\n"
        "原始arguments：\n"
        f"{json.dumps(arguments, ensure_ascii=False)}"
        "\n\n"
        "Tool真实错误：\n"
        f"{error_message}"
    )

    result = llm.chat(
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0.0,
    )

    if (
        not result.ok
        or not result.content
    ):
        return (
            None,
            "参数修复LLM调用失败",
        )

    try:
        data = json.loads(
            _clean_llm_json(
                result.content
            )
        )

    except Exception as error:
        return (
            None,
            (
                "参数修复结果解析失败："
                f"{type(error).__name__}: "
                f"{error}"
            ),
        )

    if not data.get(
        "can_retry",
        False,
    ):
        return (
            None,
            str(
                data.get(
                    "reason",
                    "参数无法可靠修复",
                )
            ),
        )

    repaired_arguments = data.get(
        "arguments"
    )

    if not isinstance(
        repaired_arguments,
        dict,
    ):
        return (
            None,
            "修复后的arguments不是字典",
        )

    if (
        repaired_arguments
        == arguments
    ):
        return (
            None,
            "修复参数与原参数完全相同",
        )

    return (
        repaired_arguments,
        str(
            data.get(
                "reason",
                "",
            )
        ),
    )

async def _call_tool_with_repair(
    client: Client,
    llm: LLMClient,
    state: AgentState,
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """
    MCP Tool统一调用入口。

    第一次：
        原参数调用

    如果失败：
        LLM基于真实错误修参

    第二次：
        修复后最多重试一次

    不允许无限重试。
    """

    # =========================
    # 第一次调用
    # =========================

    first_error = ""

    try:
        result = await client.call_tool(
            tool_name,
            arguments,
        )

        text = _extract_mcp_text(
            result
        )

        if not result.is_error:
            return {
                "ok": True,
                "result": text,
                "arguments": arguments,
                "retried": False,
                "repair_reason": "",
            }

        first_error = text

    except Exception as error:
        first_error = (
            f"{type(error).__name__}: "
            f"{error}"
        )

    # =========================
    # 自动修复参数
    # =========================

    (
        repaired_arguments,
        repair_reason,
    ) = _repair_tool_arguments(
        llm=llm,
        state=state,
        tool_name=tool_name,
        arguments=arguments,
        error_message=first_error,
    )

    if repaired_arguments is None:
        return {
            "ok": False,
            "result": "",
            "arguments": arguments,
            "retried": False,
            "first_error": first_error,
            "repair_reason": (
                repair_reason
            ),
        }

    # =========================
    # 第二次调用
    # 只允许一次
    # =========================

    try:
        retry_result = (
            await client.call_tool(
                tool_name,
                repaired_arguments,
            )
        )

        retry_text = (
            _extract_mcp_text(
                retry_result
            )
        )

        if retry_result.is_error:
            return {
                "ok": False,
                "result": "",
                "arguments": (
                    repaired_arguments
                ),
                "original_arguments": (
                    arguments
                ),
                "retried": True,
                "first_error": (
                    first_error
                ),
                "retry_error": (
                    retry_text
                ),
                "repair_reason": (
                    repair_reason
                ),
            }

        return {
            "ok": True,
            "result": retry_text,
            "arguments": (
                repaired_arguments
            ),
            "original_arguments": (
                arguments
            ),
            "retried": True,
            "first_error": (
                first_error
            ),
            "repair_reason": (
                repair_reason
            ),
        }

    except Exception as error:
        return {
            "ok": False,
            "result": "",
            "arguments": (
                repaired_arguments
            ),
            "original_arguments": (
                arguments
            ),
            "retried": True,
            "first_error": (
                first_error
            ),
            "retry_error": (
                f"{type(error).__name__}: "
                f"{error}"
            ),
            "repair_reason": (
                repair_reason
            ),
        }

def _find_rag_results(
    value: Any,
) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        results = value.get("results")

        if isinstance(results, list):
            return [
                item
                for item in results
                if isinstance(item, dict)
            ]

        for item in value.values():
            found = _find_rag_results(item)

            if found:
                return found

    elif isinstance(value, list):
        for item in value:
            found = _find_rag_results(item)

            if found:
                return found

    return []


def _parse_rag_results(
    text: str,
) -> list[dict[str, Any]]:
    if not text:
        return []

    try:
        data = json.loads(text)

    except json.JSONDecodeError:
        return []

    return _find_rag_results(data)

def create_metric_node(
    client: Client,
    llm: LLMClient,
):
    """
    实验数据 MCP Node。

    支持：
    Tool失败后自动修正参数一次。
    """

    async def metric_node(
        state: AgentState,
    ) -> dict:
        tool_name = state.get(
            "metric_tool",
            "",
        )

        arguments = state.get(
            "metric_arguments",
            {},
        )

        if not tool_name:
            return {
                "error": (
                    "metric_node没有收到"
                    "Tool名称"
                )
            }

        call_result = (
            await _call_tool_with_repair(
                client=client,
                llm=llm,
                state=state,
                tool_name=tool_name,
                arguments=arguments,
            )
        )

        if not call_result[
            "ok"
        ]:
            return {
                "error": (
                    "实验数据Tool执行失败："
                    f"{json.dumps(call_result, ensure_ascii=False)}"
                )
            }

        update: dict[
            str,
            Any,
        ] = {
            "metric_result": (
                call_result["result"]
            ),
            "tools_used": [
                tool_name
            ],
        }

        # 如果发生过自动修复，
        # 把最终参数同步回State。
        if call_result.get(
            "retried"
        ):
            update[
                "metric_arguments"
            ] = call_result[
                "arguments"
            ]

        return update

    return metric_node

def create_rag_node(
    client: Client,
):
    async def rag_node(
        state: AgentState,
    ) -> dict:
        route = state.get(
            "route",
            "rag",
        )

        user_query = state[
            "user_query"
        ]

        rag_question = (
            state.get(
                "rag_question"
            )
            or user_query
        )

        rag_queries = list(
            state.get(
                "rag_queries",
                [],
            )
        )

        top_k = state.get(
            "top_k",
            3,
        )

        cleaned_queries: list[str] = []

        for query in rag_queries:
            if not isinstance(
                query,
                str,
            ):
                continue

            query = query.strip()

            if query:
                cleaned_queries.append(
                    query
                )

        cleaned_queries = list(
            dict.fromkeys(
                cleaned_queries
            )
        )

        if not cleaned_queries:
            cleaned_queries = [
                rag_question
            ]

        if (
            route != "hybrid"
            and len(
                cleaned_queries
            ) == 1
        ):
            per_query_k = max(
                top_k,
                3,
            )

        else:
            per_query_k = 4

        merged_results: list[
            dict[str, Any]
        ] = []

        raw_evidence_parts: list[
            str
        ] = []

        query_debug: list[
            dict[str, Any]
        ] = []

        for query in cleaned_queries:
            try:
                result = (
                    await client.call_tool(
                        "search_research_docs",
                        {
                            "question": query,
                            "top_k": (
                                per_query_k
                            ),
                        },
                    )
                )

            except Exception as error:
                query_debug.append(
                    {
                        "query": query,
                        "error": (
                            f"{type(error).__name__}: "
                            f"{error}"
                        ),
                    }
                )
                continue

            text = _extract_mcp_text(
                result
            )

            if result.is_error:
                query_debug.append(
                    {
                        "query": query,
                        "error": text,
                    }
                )
                continue

            parsed_results = (
                _parse_rag_results(
                    text
                )
            )

            query_debug.append(
                {
                    "query": query,
                    "count": len(
                        parsed_results
                    ),
                }
            )

            for item in parsed_results:
                enriched = dict(item)

                enriched[
                    "matched_query"
                ] = query

                merged_results.append(
                    enriched
                )

            if text:
                raw_evidence_parts.append(
                    (
                        "检索问题：\n"
                        f"{query}\n\n"
                        "检索返回：\n"
                        f"{text}"
                    )
                )

        unique_results: list[
            dict[str, Any]
        ] = []

        seen: set[str] = set()

        for item in merged_results:
            source = str(
                item.get(
                    "source",
                    item.get(
                        "filename",
                        "",
                    ),
                )
            )

            chunk_id = str(
                item.get(
                    "chunk_id",
                    item.get(
                        "chunk",
                        "",
                    ),
                )
            )

            content = str(
                item.get(
                    "content",
                    item.get(
                        "text",
                        "",
                    ),
                )
            )

            key = (
                f"{source}::"
                f"{chunk_id}::"
                f"{content[:200]}"
            )

            if key in seen:
                continue

            seen.add(key)

            unique_results.append(
                item
            )

        final_top_k = (
            8
            if (
                route == "hybrid"
                or len(
                    cleaned_queries
                ) > 1
            )
            else top_k
        )

        unique_results = (
            unique_results[
                :final_top_k
            ]
        )

        if unique_results:
            rag_text = json.dumps(
                {
                    "ok": True,
                    "tool": (
                        "multi_query_rag"
                    ),
                    "data": {
                        "queries": (
                            cleaned_queries
                        ),
                        "count": len(
                            unique_results
                        ),
                        "results": (
                            unique_results
                        ),
                        "query_debug": (
                            query_debug
                        ),
                    },
                },
                ensure_ascii=False,
            )

        elif raw_evidence_parts:
            rag_text = "\n\n".join(
                raw_evidence_parts
            )

        else:
            rag_text = json.dumps(
                {
                    "ok": True,
                    "tool": (
                        "multi_query_rag"
                    ),
                    "data": {
                        "queries": (
                            cleaned_queries
                        ),
                        "count": 0,
                        "results": [],
                        "query_debug": (
                            query_debug
                        ),
                    },
                },
                ensure_ascii=False,
            )

        return {
            "rag_result": rag_text,
            "tools_used": [
                "search_research_docs"
            ],
        }

    return rag_node

def create_tool_executor_node(
    client: Client,
    llm: LLMClient,
):
    """
    执行Planner生成的多个MCP Tool。

    当前版本采用顺序执行：
    Action 1
        ↓
    Action 2
        ↓
    Action 3

    每一步结果都会保存，
    最后统一交给Answer Node。
    """

    async def tool_executor_node(
        state: AgentState,
    ) -> dict:
        tool_plan = state.get(
            "tool_plan",
            [],
        )

        if not tool_plan:
            return {
                "error": (
                    "Tool Executor没有收到"
                    "可执行计划"
                )
            }

        tool_results: list[
            dict[str, Any]
        ] = []

        tools_used: list[str] = []

        for index, action in enumerate(
            tool_plan,
            start=1,
        ):
            tool_name = action.get(
                "tool_name",
                "",
            )

            arguments = action.get(
                "arguments",
                {},
            )

            purpose = action.get(
                "purpose",
                "",
            )

            if not tool_name:
                tool_results.append(
                    {
                        "step": index,
                        "ok": False,
                        "tool_name": "",
                        "purpose": purpose,
                        "error": (
                            "缺少tool_name"
                        ),
                    }
                )

                continue

            call_result = (
                await _call_tool_with_repair(
                    client=client,
                    llm=llm,
                    state=state,
                    tool_name=tool_name,
                    arguments=arguments,
                )
            )

            if not call_result["ok"]:
                tool_results.append(
                    {
                        "step": index,
                        "ok": False,
                        "tool_name": tool_name,
                        "purpose": purpose,
                        "original_arguments": (
                            arguments
                        ),
                        "final_arguments": (
                            call_result.get(
                                "arguments"
                            )
                        ),
                        "retried": (
                            call_result.get(
                                "retried",
                                False,
                            )
                        ),
                        "repair_reason": (
                            call_result.get(
                                "repair_reason",
                                "",
                            )
                        ),
                        "first_error": (
                            call_result.get(
                                "first_error",
                                "",
                            )
                        ),
                        "retry_error": (
                            call_result.get(
                                "retry_error",
                                "",
                            )
                        ),
                    }
                )

                continue

            tools_used.append(
                tool_name
            )

            tool_results.append(
                {
                    "step": index,
                    "ok": True,
                    "tool_name": tool_name,
                    "purpose": purpose,
                    "original_arguments": (
                        arguments
                    ),
                    "final_arguments": (
                        call_result[
                            "arguments"
                        ]
                    ),
                    "retried": (
                        call_result.get(
                            "retried",
                            False,
                        )
                    ),
                    "repair_reason": (
                        call_result.get(
                            "repair_reason",
                            "",
                        )
                    ),
                    "result": (
                        call_result[
                            "result"
                        ]
                    ),
                }
            )

        successful_count = sum(
            1
            for item in tool_results
            if item.get("ok")
        )

        if successful_count == 0:
            return {
                "tool_results": (
                    tool_results
                ),
                "tools_used": (
                    tools_used
                ),
                "error": (
                    "Planner生成的所有Tool"
                    "均执行失败"
                ),
            }

        return {
            "tool_results": (
                tool_results
            ),
            "tools_used": (
                tools_used
            ),
        }

    return tool_executor_node

def create_answer_node(
    llm: LLMClient,
):
    """
    最终回答 Node。
    """

    def answer_node(
        state: AgentState,
    ) -> dict:
        error = state.get(
            "error",
            "",
        )

        if error:
            return {
                "final_answer": (
                    "当前任务执行失败："
                    f"{error}"
                )
            }

        metric_result = (
            state.get(
                "metric_result",
                "",
            )
        )

        rag_result = (
            state.get(
                "rag_result",
                "",
            )
        )

        tool_results = state.get(
            "tool_results",
            [],
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

        evidence_parts: list[
            str
        ] = []

        if metric_result:
            evidence_parts.append(
                "【真实实验数据】\n"
                f"{metric_result}"
            )

        if rag_result:
            evidence_parts.append(
                "【研究知识库检索结果】\n"
                f"{rag_result}"
            )

        if tool_results:
            evidence_parts.append(
                "【Planner多Tool执行结果】\n"
                + json.dumps(
                    tool_results,
                    ensure_ascii=False,
                    indent=2,
                )
            )
        
        evidence = "\n\n".join(
            evidence_parts
        )

        result = llm.chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是 SignalLab Agent。"
                        "请结合历史上下文和"
                        "当前 Tool 提供的证据"
                        "回答用户问题。"
                        "回答规则"

                        "1. 结构化实验数据负责回答："
                        "有哪些指标"
                        "指标数值是多少"
                        "哪个模型在某指标更优"
                        "哪些模型缺少某指标"
                        "是否存在协议限制"

                        "2. RAG负责回答："
                        "为什么出现这些差异"
                        "模型结构和机制"
                        "FFT、多窗口FFT、补偿机制"
                        "消融与实验解释"

                        "3. 不得因为某个模型缺少一个字段，"
                        "就说“整个Tool不支持该指标”。"

                        "4. 不得因为RAG检索0条，"
                        "就否认结构化数据库中已经存在的实验数据。"

                        "5. 多指标结果不得强行合成为"
                        "一个没有定义的“总分”。"

                        "6. 如果不同指标结论不一致，"
                        "明确说明性能存在trade-off。"

                        "7. 如果某指标不是所有模型共有，"
                        "明确列出："
                        "哪些模型有、"
                        "哪些模型没有，"
                        "不要让整个问题失败。"
                        "8. 当RAG返回多个不同来源的证据时，"
                        "应综合不同文档回答，而不是只复述排名第一的chunk。"

                        "9. 对“创新点、取舍、优势、不足”等复杂问题，"
                        "应尽量分别从："
                        "研究目标、"
                        "模型结构、"
                        "核心机制、"
                        "实验结果、"
                        "消融或局限"
                        "几个层面组织答案。"

                        "10. 如果RAG证据包含来源文件名，"
                        "回答中应说明结论来自哪类研究资料，"
                        "但不要把检索距离或内部RRF分数当成研究结论。"

                        "真实实验数值不能修改、"
                        "不能杜撰。"
                        "研究结论没有证据支持时"
                        "不能说成确定事实。"
                        "如果证据不足，"
                        "必须明确说明。"
                        "使用简洁专业的中文。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "长期历史摘要：\n"
                        f"{summary or '无'}"
                        "\n\n"
                        "最近对话：\n"
                        f"{history_text}"
                        "\n\n"
                        "当前用户问题：\n"
                        f"{state['user_query']}"
                        "\n\n"
                        "当前查询证据：\n"
                        f"{evidence or '无'}"
                    ),
                },
            ],
            temperature=0.2,
        )

        if not result.ok:
            return {
                "final_answer": (
                    "Tool 已执行，"
                    "但最终回答生成失败："
                    f"{result.error_message}"
                )
            }

        return {
            "final_answer": (
                result.content
                or "未生成最终回答。"
            )
        }

    return answer_node


def capability_node(
    state: AgentState,
) -> dict:
    """
    Agent 能力自省。
    """

    tools = state.get(
        "available_tools",
        [],
    )

    if not tools:
        return {
            "final_answer": (
                "当前没有发现可用的 "
                "MCP Tool。"
            )
        }

    tool_lines: list[str] = []

    for tool in tools:
        name = tool.get(
            "name",
            "unknown",
        )

        description = tool.get(
            "description",
            "暂无说明",
        )

        tool_lines.append(
            f"- {name}："
            f"{description}"
        )

    tool_text = "\n".join(
        tool_lines
    )

    return {
        "final_answer": (
            "目前 SignalLab Agent "
            "可以处理以下任务：\n\n"

            "1. 查询真实实验指标；\n"
            "2. 比较不同模型性能；\n"
            "3. 通过 RAG 检索研究资料；\n"
            "4. 结合实验数据和研究资料"
            "进行综合分析；\n\n"

            "当前 MCP Tool：\n"
            f"{tool_text}"
        )
    }


def fallback_node(
    state: AgentState,
) -> dict:
    """
    无法处理时执行。
    """

    error = state.get(
        "error",
        "",
    )

    if error:
        return {
            "final_answer": (
                "当前任务处理失败："
                f"{error}"
            )
        }

    return {
        "final_answer": (
            "当前 SignalLab Agent "
            "已有能力无法可靠处理"
            "这个问题。"
        )
    }