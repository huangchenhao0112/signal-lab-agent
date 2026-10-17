from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SkillDefinition:
    name: str
    description: str
    triggers: list[str] = field(
        default_factory=list
    )
    recommended_tools: list[str] = field(
        default_factory=list
    )
    instructions: str = ""