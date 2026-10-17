from agent.skills.base import (
    SkillDefinition,
)


RESEARCH_ANALYSIS_SKILL = SkillDefinition(
    name="research_analysis",
    description=(
        "用于分析NCSD-Net相关研究内容，"
        "包括模型结构、创新点、"
        "方法机制、实验设计和局限性。"
    ),
    triggers=[
        "模型结构",
        "创新点",
        "方法原理",
        "为什么这样设计",
        "有什么局限",
    ],
    recommended_tools=[
        "search_research_docs",
        "search_research_docs_multi",
    ],
    instructions=(
        "优先使用研究知识库证据回答。"
        "复杂问题拆成多个语义子查询，"
        "分别检索项目动机、模型结构、"
        "FFT参考机制、实验结果和局限。"
        "不得补充知识库中不存在的"
        "项目事实。"
    ),
)