from agent.skills.base import (
    SkillDefinition,
)

from agent.skills.model_comparison import (
    MODEL_COMPARISON_SKILL,
)

from agent.skills.research_analysis import (
    RESEARCH_ANALYSIS_SKILL,
)

from agent.skills.experiment_diagnosis import (
    EXPERIMENT_DIAGNOSIS_SKILL,
)


ALL_SKILLS: list[
    SkillDefinition
] = [
    MODEL_COMPARISON_SKILL,
    RESEARCH_ANALYSIS_SKILL,
    EXPERIMENT_DIAGNOSIS_SKILL,
]


def get_skill_catalog() -> list[dict]:
    return [
        {
            "name": skill.name,
            "description": (
                skill.description
            ),
            "triggers": (
                skill.triggers
            ),
            "recommended_tools": (
                skill.recommended_tools
            ),
            "instructions": (
                skill.instructions
            ),
        }
        for skill in ALL_SKILLS
    ]