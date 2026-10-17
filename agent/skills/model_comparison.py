from agent.skills.base import (
    SkillDefinition,
)


MODEL_COMPARISON_SKILL = SkillDefinition(
    name="model_comparison",
    description=(
        "用于多个模型之间的实验性能比较，"
        "包括指标对比、协议核对、"
        "trade-off分析和可选结果可视化。"
    ),
    triggers=[
        "比较模型",
        "哪个模型更好",
        "性能对比",
        "指标对比",
        "画对比图",
    ],
    recommended_tools=[
        "analyze_model_performance",
        "get_metric_definition",
        "get_protocol_info",
        "search_research_docs",
        "generate_metric_plot",
    ],
    instructions=(
        "优先查询真实实验数据，"
        "确认指标方向和实验协议，"
        "禁止跨协议直接比较。"
        "如果需要解释性能差异，"
        "再调用研究文档检索。"
        "如果用户要求可视化，"
        "最后调用绘图Tool。"
    ),
)