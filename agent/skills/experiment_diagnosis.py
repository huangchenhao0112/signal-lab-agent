from agent.skills.base import (
    SkillDefinition,
)


EXPERIMENT_DIAGNOSIS_SKILL = (
    SkillDefinition(
        name="experiment_diagnosis",
        description=(
            "用于诊断实验指标异常、"
            "性能trade-off以及"
            "不同实验场景下的表现差异。"
        ),
        triggers=[
            "为什么下降",
            "为什么变差",
            "为什么提升",
            "指标异常",
            "low_snr表现",
            "trade-off",
        ],
        recommended_tools=[
            "get_model_metrics",
            "analyze_model_performance",
            "get_metric_definition",
            "get_protocol_info",
            "search_research_docs",
        ],
        instructions=(
            "先查询真实实验指标，"
            "再确认指标定义和协议。"
            "只有在数据事实明确后，"
            "才使用研究文档解释原因。"
            "不得把不同协议、不同指标"
            "混为同一性能结论。"
        ),
    )
)