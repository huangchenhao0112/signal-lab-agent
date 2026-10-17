from __future__ import annotations

from typing import Callable, Any

from agent.tool_registry import (
    ToolRegistry,
)

from tools.document_tools import (
    SearchResearchDocsMultiTool,
    SearchResearchDocsTool,
)

from tools.metrics_tools import (
    analyze_model_performance,
    compare_models,
    get_experiment_config,
    get_metric_definition,
    get_model_metrics,
    get_protocol_info,
    list_model_metrics,
)

from tools.plot_tools import (
    generate_metric_plot,
)


SearchFunction = Callable[
    ...,
    list[Any],
]


def create_signal_lab_tool_registry(
    search_fn: SearchFunction,
) -> ToolRegistry:
    registry = ToolRegistry()

    # =========================
    # RAG Tools
    # =========================

    search_research_docs = (
        SearchResearchDocsTool(
            search_fn=search_fn
        )
    )

    search_research_docs_multi = (
        SearchResearchDocsMultiTool(
            search_fn=search_fn,
            rrf_k=60,
            per_source_limit=2,
        )
    )

    registry.register(
        name="search_research_docs",
        description=(
            "简单研究问题的单查询RAG检索。"
            "适合定义、单一概念、"
            "单一模块等问题。"
            "参数：question、top_k。"
        ),
        function=(
            search_research_docs
        ),
    )

    registry.register(
        name=(
            "search_research_docs_multi"
        ),
        description=(
            "复杂研究问题的多查询RAG检索主工具。"
            "接受多个不同角度的子查询，"
            "分别检索后进行RRF融合、去重和"
            "来源多样性控制。"
            "涉及创新点、结构、机制、"
            "实验结果、取舍、消融、局限、"
            "或者复杂原因分析时优先使用。"
            "参数：questions、top_k、per_query_k。"
        ),
        function=(
            search_research_docs_multi
        ),
    )

    # =========================
    # Metrics
    # =========================

    registry.register(
        name="get_model_metrics",
        description=(
            "查询单个模型在指定数据分组下"
            "的某一个真实实验指标。"
            "参数：model、metric、group。"
        ),
        function=get_model_metrics,
    )

    registry.register(
        name="list_model_metrics",
        description=(
            "列出某模型在指定group下"
            "实际存在的全部指标。"
            "参数：model、group。"
        ),
        function=list_model_metrics,
    )

    registry.register(
        name="compare_models",
        description=(
            "比较两个模型在相同group下"
            "的一个明确指标。"
            "参数：model_a、model_b、"
            "metric、group。"
        ),
        function=compare_models,
    )

    registry.register(
        name=(
            "analyze_model_performance"
        ),
        description=(
            "多模型、多指标复杂性能分析主工具。"
            "可以自动发现多个模型共同指标，"
            "给出数值、winner、缺失字段"
            "和协议信息。"
        ),
        function=(
            analyze_model_performance
        ),
    )

    registry.register(
        name="get_metric_definition",
        description=(
            "查询实验指标定义、"
            "优化方向和注意事项。"
        ),
        function=(
            get_metric_definition
        ),
    )

    registry.register(
        name="get_protocol_info",
        description=(
            "查询指标对应实验协议、"
            "数据来源和可比较性规则。"
        ),
        function=get_protocol_info,
    )

    registry.register(
        name="get_experiment_config",
        description=(
            "根据experiment_id查询"
            "真实实验配置和协议。"
        ),
        function=(
            get_experiment_config
        ),
    )

    registry.register(
        name="generate_metric_plot",
        description=(
            "根据真实结构化实验数据"
            "生成模型指标图。"
        ),
        function=(
            generate_metric_plot
        ),
    )

    return registry