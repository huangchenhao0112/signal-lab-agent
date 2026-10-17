import json
import os
import sqlite3
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel
from pydantic import Field
from typing_extensions import TypedDict

from agent.tool_registry import ToolRegistry
from uuid import uuid4
from agent.execution_policy import (
    console_confirm,
    execute_with_policy,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "data"
    / "checkpoints"
    / "langgraph.sqlite"
)

# ============================================================
# 1. 定义图状态
# ============================================================


class AgentState(TypedDict):
    """
    LangGraph中所有节点共享的状态。

    Day 22只保存当前一次请求的数据。
    Day 23再增加会话历史和Memory。
    """

    question: str

    route: str | None
    classification_reason: str

    selected_tool: str | None
    tool_arguments: dict[str, Any]

    retrieved_docs: Any
    tool_result: Any

    answer: str

    trace: list[dict[str, Any]]
    message_history: list[dict[str, str]]

# ============================================================
# 2. 大模型结构化输出
# ============================================================


class RequestClassification(BaseModel):
    """请求分类结果。"""

    route: Literal[
        "rag",
        "tool",
    ] = Field(
        description=(
            "解释性知识问题选择rag；"
            "精确数据、比较、配置或绘图选择tool"
        )
    )

    reason: str = Field(
        description="选择该路径的简短原因"
    )


class ToolDecision(BaseModel):
    """工具选择及参数。"""

    tool_name: Literal[
        "get_model_metrics",
        "compare_models",
        "get_experiment_config",
        "generate_metric_plot",
    ]

    model: str | None = None
    model_a: str | None = None
    model_b: str | None = None

    models: list[str] | None = None

    metric: str | None = None
    group: str | None = None

    experiment_id: str | None = None


# ============================================================
# 3. Graph运行结果
# ============================================================


@dataclass
class GraphRunResult:
    question: str
    route: str
    selected_tool: str | None
    answer: str
    trace: list[dict[str, Any]]
    final_state: dict[str, Any]


# ============================================================
# 4. 辅助函数
# ============================================================


def _to_serializable(
    value: Any,
) -> Any:
    """把工具返回转换成可以放入State和JSON的对象。"""

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


def _extract_message_text(
    content: Any,
) -> str:
    """兼容字符串和内容块形式的模型回复。"""

    if isinstance(content, str):
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

        return "\n".join(
            text_parts
        ).strip()

    return str(content).strip()


# ============================================================
# 5. LangGraph Agent
# ============================================================


class SignalLabGraphAgent:
    """使用LangGraph编排SignalLab Agent工作流。"""

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
                "工具调用工作流请使用deepseek-chat"
            )

        self.registry = registry

        self.decision_model = ChatDeepSeek(
        api_key=api_key,
        model=model_name,
        temperature=0.1,
        timeout=60,
        max_retries=2,
        extra_body={
            "thinking": {
                "type": "disabled",
            }
        },
    )

        self.answer_model = ChatDeepSeek(
        api_key=api_key,
        model=model_name,
        temperature=0.1,
        timeout=60,
        max_retries=2,
    )

        self.model = self.answer_model

        # 两个结构化输出模型
        self.classification_model = (
            self.decision_model.with_structured_output(
                RequestClassification
            )
        )

        self.tool_selection_model = (
            self.decision_model.with_structured_output(
                ToolDecision
            )
        )

        CHECKPOINT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._checkpoint_connection = (
            sqlite3.connect(
            str(CHECKPOINT_PATH),
            check_same_thread=False,
            )
        )

        self.checkpointer = SqliteSaver(
            self._checkpoint_connection,
        )

        self.graph = self._build_graph()

    # ========================================================
    # Node 1：问题分类
    # ========================================================

    def classify_request(
        self,
        state: AgentState,
    ) -> dict[str, Any]:
        """判断问题走RAG还是结构化工具。"""

        question = state["question"]

        classification = (
            self.classification_model.invoke(
                [
                    SystemMessage(
                        content=(
                            "你是SignalLab请求分类器。\n\n"
                            "分类规则：\n"
                            "1. 模型结构、工作原理、输入输出、"
                            "多窗口FFT作用、限制、常见错误等"
                            "解释性问题选择rag。\n"
                            "2. 查询精确实验数值选择tool。\n"
                            "3. 比较两个模型的指标选择tool。\n"
                            "4. 查询实验协议或配置选择tool。\n"
                            "5. 生成指标图选择tool。\n"
                            "6. 不要回答问题，只做分类。"
                        )
                    ),
                    HumanMessage(
                        content=question
                    ),
                ]
            )
        )

        new_trace = [
            *state["trace"],
            {
                "node": "classify_request",
                "route": classification.route,
                "reason": classification.reason,
            },
        ]

        return {
            "route": classification.route,
            "classification_reason": (
                classification.reason
            ),
            "trace": new_trace,
        }

    # ========================================================
    # Conditional Edge：分类后的路由
    # ========================================================

    @staticmethod
    def route_after_classification(
        state: AgentState,
    ) -> str:
        """根据route选择下一个节点。"""

        if state["route"] == "rag":
            return "retrieve_docs"

        return "select_tool"

    # ========================================================
    # Node 2A：RAG检索
    # ========================================================

    def retrieve_docs(
        self,
        state: AgentState,
    ) -> dict[str, Any]:
        """调用真实search_research_docs工具。"""

        arguments = {
            "question": state["question"],
            "top_k": 3,
        }

        result = self.registry.execute(
            name="search_research_docs",
            arguments=arguments,
        )

        result_data = _to_serializable(
            result
        )

        new_trace = [
            *state["trace"],
            {
                "node": "retrieve_docs",
                "tool": "search_research_docs",
                "arguments": arguments,
                "result": result_data,
            },
        ]

        return {
            "selected_tool": (
                "search_research_docs"
            ),
            "tool_arguments": arguments,
            "retrieved_docs": result_data,
            "trace": new_trace,
        }

    # ========================================================
    # Node 2B：选择结构化工具
    # ========================================================

    def select_tool(
        self,
        state: AgentState,
    ) -> dict[str, Any]:
        """根据问题选择4个结构化工具之一。"""

        decision = (
            self.tool_selection_model.invoke(
                [
                    SystemMessage(
                        content=(
                            "你是SignalLab工具选择器。\n\n"
                            "可用工具：\n"
                            "1. get_model_metrics："
                            "查询一个模型的精确指标。\n"
                            "2. compare_models："
                            "比较两个模型的同一指标。\n"
                            "3. get_experiment_config："
                            "查询实验协议和配置。\n"
                            "4. generate_metric_plot："
                            "生成指标对比图。\n\n"
                            "参数规则：\n"
                            "- get_model_metrics需要"
                            "model、metric、group。\n"
                            "- compare_models需要"
                            "model_a、model_b、metric、group。\n"
                            "- get_experiment_config需要"
                            "experiment_id。\n"
                            "- generate_metric_plot需要"
                            "models、metric、group。\n"
                            "只做工具选择和参数提取，"
                            "不要回答用户问题。"
                        )
                    ),
                    HumanMessage(
                        content=state["question"]
                    ),
                ]
            )
        )

        arguments = self._build_tool_arguments(
            decision
        )

        new_trace = [
            *state["trace"],
            {
                "node": "select_tool",
                "tool": decision.tool_name,
                "arguments": arguments,
            },
        ]

        return {
            "selected_tool": (
                decision.tool_name
            ),
            "tool_arguments": arguments,
            "trace": new_trace,
        }

    @staticmethod
    def _build_tool_arguments(
        decision: ToolDecision,
    ) -> dict[str, Any]:
        """根据工具名称构造严格匹配的参数。"""

        if (
            decision.tool_name
            == "get_model_metrics"
        ):
            required = {
                "model": decision.model,
                "metric": decision.metric,
                "group": decision.group,
            }

        elif (
            decision.tool_name
            == "compare_models"
        ):
            required = {
                "model_a": decision.model_a,
                "model_b": decision.model_b,
                "metric": decision.metric,
                "group": decision.group,
            }

        elif (
            decision.tool_name
            == "get_experiment_config"
        ):
            required = {
                "experiment_id": (
                    decision.experiment_id
                ),
            }

        elif (
            decision.tool_name
            == "generate_metric_plot"
        ):
            required = {
                "models": decision.models,
                "metric": decision.metric,
                "group": decision.group,
            }

        else:
            raise ValueError(
                f"不支持的工具：{decision.tool_name}"
            )

        missing_fields = [
            name
            for name, value in required.items()
            if value is None
            or value == ""
            or value == []
        ]

        if missing_fields:
            raise ValueError(
                "工具参数缺失："
                + ", ".join(missing_fields)
            )

        return required

    # ========================================================
    # Node 3：执行结构化工具
    # ========================================================

    def execute_tool(
        self,
        state: AgentState,
    ) -> dict[str, Any]:
        """通过真实ToolRegistry执行工具。"""

        selected_tool = state[
            "selected_tool"
        ]

        if not selected_tool:
            raise ValueError(
                "没有选择工具，无法执行"
            )

        result = execute_with_policy(
            registry=self.registry,
            name=selected_tool,
            arguments=state["tool_arguments"],
            confirm=console_confirm,
            max_retries=2,
        )

        print("\n工具执行轨迹：")

        for event in result["execution_trace"]:
            print(event)
        
        result_data = _to_serializable(
            result
        )

        new_trace = [
            *state["trace"],
            {
                "node": "execute_tool",
                "tool": selected_tool,
                "arguments": state[
                    "tool_arguments"
                ],
                "result": result_data,
            },
        ]

        return {
            "tool_result": result_data,
            "trace": new_trace,
        }

    # ========================================================
    # Node 4：生成最终回答
    # ========================================================

    def generate_answer(
        self,
        state: AgentState,
    ) -> dict[str, Any]:
        """根据RAG或工具结果生成最终答案。"""

        if state["route"] == "rag":
            evidence = state[
                "retrieved_docs"
            ]
            evidence_type = (
                "RAG知识库检索结果"
            )

        else:
            evidence = state[
                "tool_result"
            ]
            evidence_type = (
                "结构化工具执行结果"
            )

        evidence_text = json.dumps(
            evidence,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        # 读取当前问题之前的最近6条对话记录
        message_history = state.get(
            "message_history",
            [],
        )

        previous_history = message_history[:-1][-6:]

        role_name_map = {
            "user": "用户",
            "assistant": "助手",
        }

        history_text = "\n".join(
            (
                f"{role_name_map.get(str(message.get('role', '')), '未知角色')}"
                f"：{str(message.get('content', ''))}"
            )
            for message in previous_history
        )

        if history_text:
            evidence_text = (
                "以下是同一会话最近的对话历史，"
                "用于理解用户当前的追问：\n"
                f"{history_text}\n\n"
                f"以下是本次{evidence_type}：\n"
                f"{evidence_text}"
            )

        response = self.model.invoke(
            [
                SystemMessage(
                    content=(
                        "你是SignalLab项目助手。\n"
                        "只能依据提供的真实工具结果回答。\n"
                        "不得补充工具结果中不存在的数值。\n"
                        "不同实验协议和指标不能随意比较。\n"
                        "如果工具执行失败，应明确说明失败原因。\n"
                        "回答尽量简洁，并保留数据或文档来源。"
                    )
                ),
                HumanMessage(
                    content=(
                        f"用户问题：\n"
                        f"{state['question']}\n\n"
                        f"证据类型：\n"
                        f"{evidence_type}\n\n"
                        f"真实证据：\n"
                        f"{evidence_text}"
                    )
                ),
            ]
        )

        answer = _extract_message_text(
            response.content
        )

        new_trace = [
            *state["trace"],
            {
                "node": "generate_answer",
                "answer": answer,
            },
        ]

        return {
            "answer": answer,
            "message_history": [
                {
                    "role": "assistant",
                    "content": answer,
                }
            ],
            "trace": new_trace,
        }

    # ========================================================
    # 建图
    # ========================================================

    def _build_graph(self):
        """注册节点、普通边和条件边。"""

        builder = StateGraph(
            AgentState
        )

        # 注册5个节点
        builder.add_node(
            "classify_request",
            self.classify_request,
        )

        builder.add_node(
            "retrieve_docs",
            self.retrieve_docs,
        )

        builder.add_node(
            "select_tool",
            self.select_tool,
        )

        builder.add_node(
            "execute_tool",
            self.execute_tool,
        )

        builder.add_node(
            "generate_answer",
            self.generate_answer,
        )

        # 图的入口
        builder.add_edge(
            START,
            "classify_request",
        )

        # 条件边：RAG或结构化工具
        builder.add_conditional_edges(
            "classify_request",
            self.route_after_classification,
            {
                "retrieve_docs": (
                    "retrieve_docs"
                ),
                "select_tool": (
                    "select_tool"
                ),
            },
        )

        # RAG路径
        builder.add_edge(
            "retrieve_docs",
            "generate_answer",
        )

        # 结构化工具路径
        builder.add_edge(
            "select_tool",
            "execute_tool",
        )

        builder.add_edge(
            "execute_tool",
            "generate_answer",
        )

        # 工作流结束
        builder.add_edge(
            "generate_answer",
            END,
        )

        return builder.compile(
            checkpointer=self.checkpointer
        )

    # ========================================================
    # 对外运行入口
    # ========================================================

    def invoke(
        self,
        question: str,
        session_id: str | None = None,
    ) -> GraphRunResult:
        """运行完整LangGraph工作流。"""

        cleaned_question = question.strip()

        if not cleaned_question:
            raise ValueError(
                "问题不能为空"
            )

        initial_state: AgentState = {
            "question": cleaned_question,
            "route": None,
            "classification_reason": "",
            "selected_tool": None,
            "tool_arguments": {},
            "retrieved_docs": None,
            "tool_result": None,
            "answer": "",
            "trace": [],
            "message_history": [
                {
                    "role": "user",
                    "content": cleaned_question,
                }
            ],
        }

        thread_id = session_id or uuid4().hex

        final_state = self.graph.invoke(
            initial_state,
            config={"configurable": {"thread_id": thread_id}},
        )

        return GraphRunResult(
            question=cleaned_question,
            route=str(
                final_state["route"]
            ),
            selected_tool=final_state.get(
                "selected_tool"
            ),
            answer=final_state["answer"],
            trace=final_state["trace"],
            final_state=final_state,
        )

    def close(self) -> None:
        """关闭SQLite数据库连接。"""
        connection = getattr(
            self,
            "_checkpoint_connection",
            None,
        )

        if connection is not None:
            connection.close()
            
    def draw_mermaid(self) -> str:
        """返回可以复制到Markdown查看的流程图。"""

        return (
            self.graph
            .get_graph()
            .draw_mermaid()
        )