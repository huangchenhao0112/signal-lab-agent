from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.core.llm_client import LLMClient

from agent.workflow.history import (
    format_history_for_prompt,
)
from agent.workflow.state import (
    AgentState,
)


class RouteDecision(
    BaseModel
):
    route: Literal[
        "metric",
        "rag",
        "hybrid",
        "plan",
        "capability",
        "none",
    ]

    metric_tool: Literal[
    "get_model_metrics",
    "list_model_metrics",
    "compare_models",
    "analyze_model_performance",
    "get_metric_definition",
    "get_protocol_info",
    "get_experiment_config",
    "none",
] = "none"

    metric_arguments: dict[
        str,
        Any,
    ] = Field(
        default_factory=dict
    )

    rag_question: str = ""

    rag_queries: list[str] = Field(
        default_factory=list
    )

    top_k: int = 3

    reason: str = ""


def _clean_json_text(
    text: str,
) -> str:
    """
    去掉 LLM 可能返回的 Markdown
    JSON 代码块。
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


def _build_tools_prompt(
    tools: list[Any],
) -> str:
    """
    把 MCP Tool 信息转为 LLM Prompt。
    """

    items: list[str] = []

    for tool in tools:
        items.append(
            json.dumps(
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
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    return "\n\n".join(
        items
    )

def _try_direct_metric_route(
    user_query: str,
) -> dict[str, Any] | None:
    """
    只处理非常明确的：

    单模型
    +
    单指标
    +
    单分组
    +
    纯数值查询

    复杂比较、原因分析、绘图等任务
    继续交给 LLM Router 判断。
    """

    models = [
        "NCSD-Net",
        "TFA-Net",
        "HCTFR-Net",
    ]

    metrics = [
        "hit_rate",
        "weighted_l2",
        "mse",
        "psnr",
        "renyi",
        "peak_f1",
        "localization_mae",
        "fp_per_frame",
        "gfsc_at_k",
    ]

    groups = [
        "overall",
        "low_snr",
        "multi_component",
        "weak_component",
        "weak_multi_component",
        "low_snr_multi_component",
        "snr_-10_db",
        "snr_-5_db",
        "snr_0_db",
        "snr_5_db",
        "snr_10_db",
        "snr_20_db",
        "snr_30_db",
        "matlab_synthetic_low",
        "matlab_synthetic_weak",
        "matlab_motor_low",
        "matlab_motor_weak",
        "matlab_voice_low",
        "matlab_voice_weak",
    ]

    lowered = user_query.lower()

    # -------------------------
    # 1. 找出所有匹配对象
    # -------------------------

    matched_models = [
        model
        for model in models
        if model.lower() in lowered
    ]

    matched_metrics = [
        metric
        for metric in metrics
        if metric.lower() in lowered
    ]

    matched_groups = [
        group
        for group in groups
        if group.lower() in lowered
    ]

    # -------------------------
    # 2. 复杂任务禁止快速路由
    # -------------------------

    complex_keywords = [
        "比较",
        "对比",
        "谁更好",
        "为什么",
        "原因",
        "解释",
        "分析",
        "画图",
        "绘图",
        "对比图",
        "可视化",
        "生成图",
        "生成peak",
        "整体表现",
        "综合表现",
        "trade-off",
    ]

    if any(
        keyword in lowered
        for keyword in complex_keywords
    ):
        return None

    # -------------------------
    # 3. 必须严格是
    #    单模型 + 单指标 + 单group
    # -------------------------

    if (
        len(matched_models) != 1
        or len(matched_metrics) != 1
        or len(matched_groups) != 1
    ):
        return None

    model = matched_models[0]
    metric = matched_metrics[0]
    group = matched_groups[0]

    # -------------------------
    # 4. 指标知识问题
    #    不能按数值查询处理
    # -------------------------

    definition_words = [
        "什么是",
        "什么意思",
        "含义",
        "定义",
        "怎么计算",
        "如何计算",
    ]

    if any(
        word in user_query
        for word in definition_words
    ):
        return None

    # -------------------------
    # 5. 真正简单、确定的
    #    单指标查询
    # -------------------------

    return {
        "route": "metric",
        "route_reason": (
            "检测到明确的单模型、"
            "单指标、单分组实验数据查询"
        ),
        "metric_tool": (
            "get_model_metrics"
        ),
        "metric_arguments": {
            "model": model,
            "metric": metric,
            "group": group,
        },
        "rag_question": "",
        "rag_queries": [],
        "top_k": 3,
    }

def create_router_node(
    llm: LLMClient,
    tools: list[Any],
):
    """
    创建 LangGraph Router Node。
    """

    tools_prompt = _build_tools_prompt(
        tools
    )

    def router_node(
        state: AgentState,
    ) -> dict:
        user_query = state[
            "user_query"
        ]

        direct_metric_route = (
        _try_direct_metric_route(
            user_query
            )
        )

        if (
            direct_metric_route
            is not None
        ):
            return (
                direct_metric_route
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

        system_prompt = f"""
你是 SignalLab Agent 的统一路由器。

你的任务不是直接回答问题，
而是结合历史上下文和当前问题，
判断应该执行什么任务。

==============================
长期历史摘要
==============================

{summary or "暂无历史摘要。"}

==============================
最近对话
==============================

{history_text}

==============================
当前 MCP Tool
==============================

{tools_prompt}

==============================
路由类型
==============================

route 只能选择：

1. metric

用户主要查询结构化实验数据，例如：

- 单个模型的某个指标；
- 多个模型性能比较；
- 某个group下有哪些指标；
- 多模型、多指标综合比较；
- 指标定义；
- 实验协议和可比较性规则。

2. rag

用户主要询问研究知识和文本内容，例如：

- 算法原理；
- 模型结构；
- FFT参考证据；
- 多窗口FFT；
- 模块设计；
- 方法机制；
- 实验设计说明；
- 论文或研究文档中的解释性内容。

如果只是询问某个实验指标的真实数值，
优先使用metric，而不是rag。

3. hybrid

用户的问题既需要真实实验数据，
又需要研究知识解释。

典型情况包括：

- “NCSD-Net在low_snr条件下和
  TFA-Net、HCTFR-Net谁效果好，为什么？”

- “NCSD-Net的Peak F1比TFA-Net高，
  这和FFT参考证据有什么关系？”

- “这些模型在低信噪比下各指标表现如何，
  为什么会出现这种trade-off？”

这类问题必须：

先使用结构化实验数据Tool得到真实结果，
再使用RAG检索研究文档解释原因。

4. plan

用户的一个请求需要连续执行
两个或以上不同类型的Tool任务。

典型情况：

- 比较模型 + 解释原因 + 画图；
- 查询实验数据 + 查询协议 + 画图；
- 多个不同Tool共同完成一个目标。

例如：

“比较NCSD-Net、TFA-Net、
HCTFR-Net在low_snr下的表现，
解释原因，并画peak_f1对比图。”

应选择：

route = plan

注意：

仅仅“比较并解释原因”
仍然可以使用hybrid。

只有任务包含更多独立操作，
例如绘图、配置查询、
多个不同Tool任务组合时，
才使用plan。

5. capability

用户询问Agent自己能做什么、
支持哪些功能、有哪些Tool。

6. none

当前已有能力无法可靠处理。

==============================
RAG Query Planning 规则
==============================

不要把复杂研究问题只改写成一个很长的Query。

普通、单一知识问题：
可以只生成1个rag_query。

复杂研究问题：
应拆成3~5个语义互补的rag_queries。

以下问题属于复杂研究问题：

- 创新点
- 方法取舍
- 为什么性能不同
- 模型优势和不足
- 多模型机制比较
- FFT作用
- 消融实验解释
- 多指标trade-off
- 论文整体方法分析

例如用户问：

“NCSD-Net做了哪些取舍，
创新点是什么？”

应生成类似：

[
  "NCSD-Net 项目定位 研究动机 核心创新",
  "NCSD-Net 模型结构 核心模块 机制设计",
  "NCSD-Net 多窗口FFT FFT参考证据 频谱补偿",
  "NCSD-Net 实验结果 指标trade-off 性能取舍",
  "NCSD-Net 消融实验 局限 不足"
]

这些Query必须：

1. 保留用户真正关注的模型或对象；
2. 从不同证据角度拆分；
3. 不要五个Query表达完全相同的意思；
4. 不要凭空加入数据库中不存在的模型名；
5. 复杂问题优先覆盖：
   方法定位、模型结构、核心机制、
   实验结果、取舍、消融或局限。

对于复杂rag或hybrid问题：

top_k = 8

普通单一概念问题：

top_k = 3

==============================
数据工具选择规则
==============================

1. 单模型 + 明确单指标：
   get_model_metrics

2. 用户询问某模型在某group下
   “还有哪些指标”“有哪些指标”：
   list_model_metrics

3. 两个模型 + 明确只有一个指标：
   compare_models

4. 以下情况优先使用
   analyze_model_performance：

   - 比较三个及以上模型
   - 用户说“整体谁更好”
   - 用户没有指定metric
   - 用户问“除了hit_rate其他指标呢”
   - 用户要求多指标比较
   - 用户询问多个模型在某group下表现

5. 用户询问指标定义：
   get_metric_definition

6. 用户询问实验协议、
   可比较性规则：
   get_protocol_info

==============================
RAG 检索规则
==============================

1. 普通 rag 问题：
   top_k 默认使用 3。

2. hybrid 复杂问题：
   top_k 应优先使用 8。

3. 如果问题同时涉及：
   - 多个模型；
   - 多个指标；
   - 原因分析；
   - 模型结构；
   - FFT机制；
   - 实验trade-off；

   应设置：
   top_k = 8

4. 不要因为文档数量是8，
   就机械理解成必须返回8条；
   top_k=8表示最多检索更多候选chunk，
   最终回答仍然只使用真正相关内容。

==============================
上下文规则
==============================

用户可能使用省略表达：
“那其他指标呢？”
“那 HCTFR-Net 呢？”
“为什么？”
“换成 low_snr 呢？”
“它呢？”

此时必须结合：

长期历史摘要
+
最近对话

理解这些代词和省略内容。

如果历史中已经确定模型、指标、
数据分组或比较对象，
当前用户没有重新指定，
应该优先继承历史上下文。

不要把：

“那其他指标呢？”
错误理解成指标定义知识查询。

上下文继承不能覆盖当前用户的明确新意图。

如果当前问题已经明确提出新的研究主题，例如：

“NCSD-Net有哪些创新点？”
“NCSD-Net做了哪些取舍？”
“介绍NCSD-Net的模型结构”

则应以当前问题为主。

历史中的模型比较对象、
数据分组、
上一轮指标，
只能作为辅助上下文，
不能强制继承到新的独立研究问题。

只有用户使用明显省略表达时，例如：

“那其他指标呢？”
“那HCTFR-Net呢？”
“为什么？”
“换成low_snr呢？”

才强继承上一轮上下文。
==============================
Tool 规则
==============================

查询单个模型指标：

metric_tool =
get_model_metrics

比较两个模型：

metric_tool =
compare_models

metric_arguments 必须严格符合
MCP Tool input_schema。

不能自行编造实验数据。

hybrid 必须同时提供：

metric_tool
metric_arguments
rag_question

capability 不需要 metric_arguments。

只返回合法 JSON，
不要返回 Markdown。

返回格式：

{{
    "route": "metric | rag | hybrid | plan | capability | none",
    "metric_tool": "工具名或none",
    "metric_arguments": {{}},
    "rag_question": "",
    "rag_queries": [],
    "top_k": 3,
    "reason": ""
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
                "route": "none",
                "error": (
                    "Router LLM 调用失败："
                    f"{result.error_type} - "
                    f"{result.error_message}"
                ),
            }

        if not result.content:
            return {
                "route": "none",
                "error": (
                    "Router LLM 返回为空"
                ),
            }

        try:
            raw_data = json.loads(
                _clean_json_text(
                    result.content
                )
            )

            decision = (
                RouteDecision(
                    **raw_data
                )
            )

        except Exception as error:
            return {
                "route": "none",
                "error": (
                    "Router 结果解析失败："
                    f"{type(error).__name__}: "
                    f"{error}"
                ),
            }

        return {
            "route": (
                decision.route
            ),
            "route_reason": (
                decision.reason
            ),
            "metric_tool": (
                decision.metric_tool
            ),
            "metric_arguments": (
                decision.metric_arguments
            ),
            "rag_question": (
                decision.rag_question
            ),
            "rag_queries": (
                decision.rag_queries
            ),
            "top_k": (
                decision.top_k
            ),
        }

    return router_node