from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    get_type_hints,
)

from pydantic import (
    TypeAdapter,
    ValidationError,
)


ToolCallable = Callable[..., Any]


@dataclass(frozen=True)
class ToolDefinition:
    """
    一个已注册工具的定义。
    """

    name: str
    description: str
    function: ToolCallable


@dataclass
class ToolExecutionResult:
    """
    ToolRegistry执行工具后的统一结果。
    """

    ok: bool
    tool_name: str
    output: Any = None
    error_type: str | None = None
    error_message: str | None = None

    def to_observation(self) -> str:
        """
        转成可放回Agent上下文的Observation。
        """

        payload = {
            "ok": self.ok,
            "tool_name": self.tool_name,
            "output": self.output,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }

        return json.dumps(
            payload,
            ensure_ascii=False,
            default=str,
        )


class ToolRegistry:
    """
    SignalLab Agent工具注册表。

    职责：

    1. 注册工具；
    2. 列出工具；
    3. 生成工具描述；
    4. 检查工具是否存在；
    5. 校验工具参数；
    6. 执行工具；
    7. 返回统一结果。
    """

    def __init__(self) -> None:
        self._tools: dict[
            str,
            ToolDefinition,
        ] = {}

    def register(
        self,
        name: str,
        description: str,
        function: ToolCallable,
    ) -> None:
        """
        注册一个Python函数或可调用对象。
        """

        cleaned_name = name.strip()

        if not cleaned_name:
            raise ValueError(
                "工具名称不能为空"
            )

        if not description.strip():
            raise ValueError(
                f"工具{cleaned_name}缺少描述"
            )

        if not callable(function):
            raise TypeError(
                f"工具{cleaned_name}必须可调用"
            )

        if cleaned_name in self._tools:
            raise ValueError(
                f"工具{cleaned_name}已注册"
            )

        self._tools[cleaned_name] = (
            ToolDefinition(
                name=cleaned_name,
                description=description.strip(),
                function=function,
            )
        )

    def has_tool(
        self,
        name: str,
    ) -> bool:
        """
        检查工具是否存在。
        """

        return name in self._tools

    def list_tools(
        self,
    ) -> list[str]:
        """
        返回全部工具名称。
        """

        return list(self._tools.keys())

    def describe_tools(self) -> str:
        """
        生成供LLM阅读的工具说明。
        """

        if not self._tools:
            return "当前没有可用工具。"

        descriptions: list[str] = []

        for tool in self._tools.values():
            signature = inspect.signature(
                tool.function
            )

            descriptions.append(
                f"- {tool.name}{signature}: "
                f"{tool.description}"
            )

        return "\n".join(descriptions)

    def execute(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> ToolExecutionResult:
        """
        校验参数并执行工具。
        """

        tool = self._tools.get(name)

        if tool is None:
            available = ", ".join(
                self.list_tools()
            )

            return ToolExecutionResult(
                ok=False,
                tool_name=name,
                error_type="tool_not_found",
                error_message=(
                    f"不存在工具{name}。"
                    f"可用工具：{available or '无'}"
                ),
            )

        if not isinstance(
            arguments,
            dict,
        ):
            return ToolExecutionResult(
                ok=False,
                tool_name=name,
                error_type="invalid_arguments",
                error_message=(
                    "工具参数必须是JSON对象"
                ),
            )

        validation_result = (
            self._validate_arguments(
                tool.function,
                arguments,
            )
        )

        if not validation_result["ok"]:
            return ToolExecutionResult(
                ok=False,
                tool_name=name,
                error_type="invalid_arguments",
                error_message=(
                    validation_result["error"]
                ),
            )

        validated_arguments = (
            validation_result["arguments"]
        )

        try:
            output = tool.function(
                **validated_arguments
            )

            return ToolExecutionResult(
                ok=True,
                tool_name=name,
                output=output,
            )

        except Exception as error:
            return ToolExecutionResult(
                ok=False,
                tool_name=name,
                error_type="tool_execution_error",
                error_message=(
                    f"{type(error).__name__}: "
                    f"{error}"
                ),
            )

    def _validate_arguments(
        self,
        function: ToolCallable,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """
        根据函数签名和类型注解校验参数。
        """

        signature = inspect.signature(
            function
        )

        try:
            bound_arguments = signature.bind(
                **arguments
            )

            bound_arguments.apply_defaults()

        except TypeError as error:
            return {
                "ok": False,
                "arguments": {},
                "error": str(error),
            }

        hint_target = function

        if not inspect.isfunction(function):
            hint_target = function.__call__

        try:
            type_hints = get_type_hints(
                hint_target
            )
        except Exception:
            type_hints = {}

        validated: dict[str, Any] = {}

        try:
            for parameter_name, value in (
                bound_arguments.arguments.items()
            ):
                annotation = type_hints.get(
                    parameter_name
                )

                if annotation is None:
                    validated[
                        parameter_name
                    ] = value
                    continue

                adapter = TypeAdapter(
                    annotation
                )

                validated[
                    parameter_name
                ] = adapter.validate_python(
                    value
                )

        except ValidationError as error:
            return {
                "ok": False,
                "arguments": {},
                "error": str(error),
            }

        return {
            "ok": True,
            "arguments": validated,
            "error": None,
        }