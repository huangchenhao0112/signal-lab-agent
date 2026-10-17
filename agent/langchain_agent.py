#不重写工具，只把现有 ToolRegistry 适配成 LangChain 工具
import json
import os
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel
from pydantic import Field

from agent.tool_registry import ToolRegistry


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# 五个工具的输入Schema
# ============================================================


class GetModelMetricsInput(BaseModel):
    model: str = Field(
        description="模型名称，例如NCSD-Net、TFA-Net"
    )

    metric: str = Field(
        description="指标名称，例如hit_rate、peak_f1"
    )

    group: str = Field(
        description="实验分组，例如Overall、Low SNR"
    )


class CompareModelsInput(BaseModel):
    model_a: str = Field(
        description="第一个模型名称"
    )

    model_b: str = Field(
        description="第二个模型名称"
    )

    metric: str = Field(
        description="需要比较的指标"
    )

    group: str = Field(
        description="实验分组"
    )


class SearchResearchDocsInput(BaseModel):
    question: str = Field(
        description="需要从NCSD-Net知识库检索的问题"
    )

    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="返回的相关文档块数量",
    )


class GetExperimentConfigInput(BaseModel):
    experiment_id: str = Field(
        description=(
            "实验协议ID，例如FROZEN_SUPPORT_V1、"
            "FROZEN_STRICT_MAIN_V1"
        )
    )


class GenerateMetricPlotInput(BaseModel):
    models: list[str] = Field(
        description="需要绘图的模型名称列表"
    )

    metric: str | None = Field(
        default=None,
        description="指标名称；不填写时生成全部指标"
)

    group: str = Field(
        description="实验分组"
    )


# ============================================================
# Agent运行结果
# ============================================================


@dataclass
class AgentRunResult:
    answer: str
    trace: list[dict[str, Any]]


# ============================================================
# ToolRegistry结果转换
# ============================================================


def _to_serializable(
    value: Any,
) -> Any:
    """将不同类型的结果转换成可JSON序列化对象。"""

    if is_dataclass(value):
        return asdict(value)

    if hasattr(value, "model_dump"):
        return value.model_dump()

    if isinstance(value, dict):
        return {
            str(key): _to_serializable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            _to_serializable(item)
            for item in value
        ]

    if hasattr(value, "__dict__"):
        return {
            key: _to_serializable(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }

    return value


def _execute_registry_tool(
    registry: ToolRegistry,
    tool_name: str,
    arguments: dict[str, Any],
) -> str:
    """
    通过项目原来的ToolRegistry执行工具。

    LangChain Tool的输出需要交回大模型，
    因此统一转换成JSON字符串。
    """

    result = registry.execute(
        name=tool_name,
        arguments=arguments,
    )

    result_data = _to_serializable(result)

    return json.dumps(
        result_data,
        ensure_ascii=False,
        default=str,
    )


# ============================================================
# 将5个真实工具转换成LangChain Tool
# ============================================================


def create_langchain_tools(
    registry: ToolRegistry,
) -> list[StructuredTool]:
    """将原来的5个工具适配成LangChain工具。"""

    def get_model_metrics(
        model: str,
        metric: str,
        group: str,
    ) -> str:
        return _execute_registry_tool(
            registry=registry,
            tool_name="get_model_metrics",
            arguments={
                "model": model,
                "metric": metric,
                "group": group,
            },
        )

    def compare_models(
        model_a: str,
        model_b: str,
        metric: str,
        group: str,
    ) -> str:
        return _execute_registry_tool(
            registry=registry,
            tool_name="compare_models",
            arguments={
                "model_a": model_a,
                "model_b": model_b,
                "metric": metric,
                "group": group,
            },
        )

    def search_research_docs(
        question: str,
        top_k: int = 3,
    ) -> str:
        return _execute_registry_tool(
            registry=registry,
            tool_name="search_research_docs",
            arguments={
                "question": question,
                "top_k": top_k,
            },
        )

    def get_experiment_config(
        experiment_id: str,
    ) -> str:
        return _execute_registry_tool(
            registry=registry,
            tool_name="get_experiment_config",
            arguments={
                "experiment_id": experiment_id,
            },
        )

    def generate_metric_plot(
        models: list[str],
        metric: str,
        group: str,
    ) -> str:
        return _execute_registry_tool(
            registry=registry,
            tool_name="generate_metric_plot",
            arguments={
                "models": models,
                "metric": metric,
                "group": group,
            },
        )

    return [
        StructuredTool.from_function(
            func=get_model_metrics,
            name="get_model_metrics",
            description=(
                "查询某个模型在指定指标和实验分组下的"
                "真实实验数值。查询精确数值时使用。"
            ),
            args_schema=GetModelMetricsInput,
        ),
        StructuredTool.from_function(
            func=compare_models,
            name="compare_models",
            description=(
                "在相同指标和相同分组下比较两个模型。"
                "不能使用本工具进行跨协议比较。"
            ),
            args_schema=CompareModelsInput,
        ),
        StructuredTool.from_function(
            func=search_research_docs,
            name="search_research_docs",
            description=(
                "从NCSD-Net知识库检索模型结构、原理、"
                "输入输出、FFT机制、限制和常见错误。"
            ),
            args_schema=SearchResearchDocsInput,
        ),
        StructuredTool.from_function(
            func=get_experiment_config,
            name="get_experiment_config",
            description=(
                "查询实验协议配置、实验规则、检查点角色"
                "以及指标可比性。"
            ),
            args_schema=GetExperimentConfigInput,
        ),
        StructuredTool.from_function(
            func=generate_metric_plot,
            name="generate_metric_plot",
            description=(
                "根据真实结构化实验数据生成模型指标对比图。"
            ),
            args_schema=GenerateMetricPlotInput,
        ),
    ]


# ============================================================
# LangChain Agent
# ============================================================


class SignalLabLangChainAgent:
    """使用LangChain和DeepSeek实现的SignalLab Agent。"""

    def __init__(
        self,
        registry: ToolRegistry,
    ) -> None:
        load_dotenv(
            PROJECT_ROOT / ".env"
        )

        api_key = os.getenv(
            "LLM_API_KEY"
        )

        model_name = os.getenv(
            "LLM_MODEL",
            "deepseek-chat",
        )

        if not api_key:
            raise ValueError(
                ".env中缺少LLM_API_KEY"
            )

        if model_name == "deepseek-reasoner":
            raise ValueError(
                "当前Agent工具调用请使用deepseek-chat，"
                "不要使用deepseek-reasoner"
            )

        self.registry = registry

        self.model = ChatDeepSeek(
            api_key=api_key,
            model=model_name,
            temperature=0.1,
            timeout=60,
            max_retries=2,
        )

        self.tools = create_langchain_tools(
            registry
        )

        self.agent = create_agent(
            model=self.model,
            tools=self.tools,
            system_prompt=(
                "你是SignalLab Agent，负责回答"
                "NCSD-Net神经时频分析项目问题。\n\n"

                "工具选择规则：\n"
                "1. 查询精确实验数值时，使用"
                "get_model_metrics。\n"
                "2. 比较两个模型时，使用"
                "compare_models。\n"
                "3. 查询模型结构、原理、输入输出、"
                "多窗口FFT作用或限制时，使用"
                "search_research_docs。\n"
                "4. 查询实验协议、阈值、评测规则或"
                "检查点用途时，使用"
                "get_experiment_config。\n"
                "5. 用户明确要求绘图时，使用"
                "generate_metric_plot。\n"
                "6. 不同实验协议、阈值和指标不能"
                "随意比较。\n"
                "7. 工具没有返回的数据不得编造。\n"
                "8. 如果工具执行失败，应明确告诉用户"
                "失败原因。\n"
                "9. 最终回答需要说明数据或文档来源。"
            ),
        )

    def invoke(
        self,
        user_input: str,
    ) -> AgentRunResult:
        """运行Agent并提取工具调用轨迹。"""

        cleaned_input = user_input.strip()

        if not cleaned_input:
            raise ValueError(
                "用户问题不能为空"
            )

        result = self.agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": cleaned_input,
                    }
                ]
            }
        )

        messages = result.get(
            "messages",
            [],
        )

        trace = self._extract_trace(
            messages
        )

        answer = self._extract_answer(
            messages
        )

        return AgentRunResult(
            answer=answer,
            trace=trace,
        )

    @staticmethod
    def _extract_trace(
        messages: list[Any],
    ) -> list[dict[str, Any]]:
        """从LangChain消息中提取工具调用和返回轨迹。"""

        trace: list[dict[str, Any]] = []

        for message in messages:
            if isinstance(
                message,
                AIMessage,
            ):
                tool_calls = (
                    message.tool_calls
                    or []
                )

                for tool_call in tool_calls:
                    trace.append(
                        {
                            "type": "tool_call",
                            "tool": tool_call.get(
                                "name"
                            ),
                            "arguments": tool_call.get(
                                "args",
                                {},
                            ),
                            "call_id": tool_call.get(
                                "id"
                            ),
                        }
                    )

            if isinstance(
                message,
                ToolMessage,
            ):
                trace.append(
                    {
                        "type": "tool_result",
                        "tool": message.name,
                        "content": message.content,
                        "call_id": (
                            message.tool_call_id
                        ),
                    }
                )

        return trace

    @staticmethod
    def _extract_answer(
        messages: list[Any],
    ) -> str:
        """从最后一条AI消息提取最终答案。"""

        for message in reversed(messages):
            if not isinstance(
                message,
                AIMessage,
            ):
                continue

            if message.tool_calls:
                continue

            content = message.content

            if isinstance(content, str):
                if content.strip():
                    return content.strip()

            if isinstance(content, list):
                text_parts: list[str] = []

                for block in content:
                    if isinstance(block, str):
                        text_parts.append(block)

                    elif isinstance(block, dict):
                        text = block.get("text")

                        if text:
                            text_parts.append(
                                str(text)
                            )

                joined_text = "\n".join(
                    text_parts
                ).strip()

                if joined_text:
                    return joined_text

        return "模型没有返回最终回答"