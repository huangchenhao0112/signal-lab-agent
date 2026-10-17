from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Protocol,
)

from agent.tool_registry import (
    ToolRegistry,
)


class LLMResultLike(Protocol):
    ok: bool
    content: str | None
    error_type: str | None
    error_message: str | None


class LLMClientLike(Protocol):
    def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.2,
    ) -> LLMResultLike:
        ...


ToolFunction = Callable[..., Any]


@dataclass
class AgentDecision:
    action: str
    thought: str = ""
    tool_name: str | None = None
    tool_input: dict[str, Any] = field(
        default_factory=dict
    )
    answer: str | None = None


@dataclass
class AgentStep:
    step_number: int
    thought: str
    action: str
    tool_name: str | None = None
    tool_input: dict[str, Any] = field(
        default_factory=dict
    )
    observation: str | None = None
    tool_ok: bool | None = None


@dataclass
class AgentResult:
    ok: bool
    answer: str | None
    steps: list[AgentStep]
    tool_call_count: int
    error_type: str | None = None
    error_message: str | None = None


class BasicAgent:
    """
    原生ReAct Agent。

    支持：

    - ToolRegistry；
    - 工具参数校验；
    - 工具不存在处理；
    - 最大Agent步数；
    - 最大工具调用次数；
    - Action/Observation记录。
    """

    def __init__(
        self,
        llm_client: LLMClientLike,
        tools: dict[
            str,
            ToolFunction,
        ] | None = None,
        max_steps: int = 5,
        *,
        tool_registry: (
            ToolRegistry | None
        ) = None,
        max_tool_calls: int = 5,
    ) -> None:
        if max_steps <= 0:
            raise ValueError(
                "max_steps必须大于0"
            )

        if max_tool_calls <= 0:
            raise ValueError(
                "max_tool_calls必须大于0"
            )

        if (
            tools is not None
            and tool_registry is not None
        ):
            raise ValueError(
                "tools和tool_registry"
                "不能同时传入"
            )

        self.llm_client = llm_client
        self.max_steps = max_steps
        self.max_tool_calls = (
            max_tool_calls
        )

        if tool_registry is not None:
            self.tool_registry = (
                tool_registry
            )
        else:
            self.tool_registry = (
                self._registry_from_tools(
                    tools or {}
                )
            )

    def run(
        self,
        question: str,
    ) -> AgentResult:
        cleaned_question = question.strip()

        if not cleaned_question:
            return AgentResult(
                ok=False,
                answer=None,
                steps=[],
                tool_call_count=0,
                error_type="invalid_question",
                error_message=(
                    "用户问题不能为空"
                ),
            )

        steps: list[AgentStep] = []
        history: list[dict[str, Any]] = []
        tool_call_count = 0

        for step_number in range(
            1,
            self.max_steps + 1,
        ):
            messages = self._build_messages(
                question=cleaned_question,
                history=history,
            )

            llm_result = (
                self.llm_client.chat(
                    messages=messages,
                    temperature=0.0,
                )
            )

            if not llm_result.ok:
                return AgentResult(
                    ok=False,
                    answer=None,
                    steps=steps,
                    tool_call_count=(
                        tool_call_count
                    ),
                    error_type=(
                        getattr(
                            llm_result,
                            "error_type",
                            None,
                        )
                        or "llm_error"
                    ),
                    error_message=(
                        getattr(
                            llm_result,
                            "error_message",
                            None,
                        )
                        or "LLM调用失败"
                    ),
                )

            raw_content = (
                llm_result.content or ""
            ).strip()

            try:
                decision = (
                    self._parse_decision(
                        raw_content
                    )
                )

            except ValueError as error:
                return AgentResult(
                    ok=False,
                    answer=None,
                    steps=steps,
                    tool_call_count=(
                        tool_call_count
                    ),
                    error_type=(
                        "decision_parse_error"
                    ),
                    error_message=str(error),
                )

            current_step = AgentStep(
                step_number=step_number,
                thought=decision.thought,
                action=decision.action,
                tool_name=decision.tool_name,
                tool_input=(
                    decision.tool_input
                ),
            )

            if decision.action == "finish":
                answer = (
                    decision.answer or ""
                ).strip()

                if not answer:
                    return AgentResult(
                        ok=False,
                        answer=None,
                        steps=steps,
                        tool_call_count=(
                            tool_call_count
                        ),
                        error_type="empty_answer",
                        error_message=(
                            "LLM选择finish，"
                            "但没有提供answer"
                        ),
                    )

                steps.append(current_step)

                return AgentResult(
                    ok=True,
                    answer=answer,
                    steps=steps,
                    tool_call_count=(
                        tool_call_count
                    ),
                )

            if decision.action != "tool":
                return AgentResult(
                    ok=False,
                    answer=None,
                    steps=steps,
                    tool_call_count=(
                        tool_call_count
                    ),
                    error_type="invalid_action",
                    error_message=(
                        "action只能是"
                        "tool或finish"
                    ),
                )

            if (
                tool_call_count
                >= self.max_tool_calls
            ):
                return AgentResult(
                    ok=False,
                    answer=None,
                    steps=steps,
                    tool_call_count=(
                        tool_call_count
                    ),
                    error_type=(
                        "max_tool_calls_exceeded"
                    ),
                    error_message=(
                        "Agent达到最大工具"
                        f"调用次数"
                        f"{self.max_tool_calls}"
                    ),
                )

            tool_call_count += 1

            execution_result = (
                self.tool_registry.execute(
                    name=(
                        decision.tool_name
                        or ""
                    ),
                    arguments=(
                        decision.tool_input
                    ),
                )
            )

            observation = (
                execution_result
                .to_observation()
            )

            current_step.observation = (
                observation
            )

            current_step.tool_ok = (
                execution_result.ok
            )

            steps.append(current_step)

            history.append(
                {
                    "step": step_number,
                    "thought": (
                        decision.thought
                    ),
                    "action": "tool",
                    "tool_name": (
                        decision.tool_name
                    ),
                    "tool_input": (
                        decision.tool_input
                    ),
                    "observation": (
                        observation
                    ),
                }
            )

        return AgentResult(
            ok=False,
            answer=None,
            steps=steps,
            tool_call_count=tool_call_count,
            error_type="max_steps_exceeded",
            error_message=(
                "Agent达到最大执行步数"
                f"{self.max_steps}，"
                "仍未得到最终答案"
            ),
        )

    def _registry_from_tools(
        self,
        tools: dict[
            str,
            ToolFunction,
        ],
    ) -> ToolRegistry:
        """
        兼容Day18的tools字典。
        """

        registry = ToolRegistry()

        for name, function in tools.items():
            description = (
                inspect.getdoc(function)
                or f"执行{name}工具"
            )

            registry.register(
                name=name,
                description=description,
                function=function,
            )

        return registry

    def _build_messages(
        self,
        question: str,
        history: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        tools_description = (
            self.tool_registry
            .describe_tools()
        )

        if history:
            history_text = json.dumps(
                history,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        else:
            history_text = (
                "尚未调用任何工具。"
            )

        system_prompt = f"""
你是SignalLab Agent的任务决策模块。

可用工具：

{tools_description}

你必须只输出一个合法JSON对象，
不能输出Markdown代码块或额外说明。

调用工具：

{{
  "thought": "简短决策摘要",
  "action": "tool",
  "tool_name": "工具名称",
  "tool_input": {{
    "参数名": "参数值"
  }}
}}

完成回答：

{{
  "thought": "简短决策摘要",
  "action": "finish",
  "answer": "最终回答"
}}

规则：

1. action只能是tool或finish；
2. tool_name必须来自可用工具；
3. tool_input必须是JSON对象；
4. 不得虚构工具结果；
5. 工具失败时，应根据Observation修正参数；
6. 不要使用完全相同参数重复调用失败工具；
7. 信息充分后必须使用finish；
8. thought只写简短决策摘要。
""".strip()

        user_prompt = f"""
用户问题：

{question}

此前Action和Observation：

{history_text}

请决定下一步。
""".strip()

        return [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

    def _parse_decision(
        self,
        raw_content: str,
    ) -> AgentDecision:
        cleaned = self._remove_code_fence(
            raw_content
        )

        try:
            data = json.loads(cleaned)

        except json.JSONDecodeError as error:
            raise ValueError(
                "LLM没有返回合法JSON："
                f"{error.msg}"
            ) from error

        if not isinstance(data, dict):
            raise ValueError(
                "LLM输出最外层必须"
                "是JSON对象"
            )

        action = data.get("action")

        if action not in {
            "tool",
            "finish",
        }:
            raise ValueError(
                "action必须是"
                "tool或finish"
            )

        thought = str(
            data.get("thought", "")
        ).strip()

        if action == "finish":
            return AgentDecision(
                action="finish",
                thought=thought,
                answer=data.get("answer"),
            )

        tool_name = data.get("tool_name")
        tool_input = data.get(
            "tool_input",
            {},
        )

        if not isinstance(
            tool_name,
            str,
        ) or not tool_name.strip():
            raise ValueError(
                "调用工具时必须提供"
                "tool_name"
            )

        if not isinstance(
            tool_input,
            dict,
        ):
            raise ValueError(
                "tool_input必须是"
                "JSON对象"
            )

        return AgentDecision(
            action="tool",
            thought=thought,
            tool_name=tool_name.strip(),
            tool_input=tool_input,
        )

    @staticmethod
    def _remove_code_fence(
        content: str,
    ) -> str:
        cleaned = content.strip()

        if not cleaned.startswith("```"):
            return cleaned

        lines = cleaned.splitlines()

        if lines:
            lines = lines[1:]

        if (
            lines
            and lines[-1].strip() == "```"
        ):
            lines = lines[:-1]

        return "\n".join(lines).strip()