import json
import sys
from pathlib import Path
from typing import Any
from dataclasses import asdict, is_dataclass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from agent.tool_factory import (
    create_signal_lab_tool_registry,
)
from app.core.config import get_settings

from app.rag.embedding_model import (
    EmbeddingModel,
)

from app.rag.vector_store import (
    VectorStore,
)

from app.rag.bm25_retriever import (
    BM25Retriever,
)

from app.rag.hybrid_retriever import (
    HybridRetriever,
)

from dataclasses import (
    asdict,
    is_dataclass,
)
from typing import Any


def _to_jsonable(
    value: Any,
) -> Any:
    if is_dataclass(value):
        return _to_jsonable(
            asdict(value)
        )

    if hasattr(
        value,
        "model_dump",
    ):
        return _to_jsonable(
            value.model_dump()
        )

    if isinstance(
        value,
        dict,
    ):
        return {
            str(key): _to_jsonable(
                item
            )
            for key, item in value.items()
        }

    if isinstance(
        value,
        (list, tuple),
    ):
        return [
            _to_jsonable(item)
            for item in value
        ]

    if hasattr(
        value,
        "__dict__",
    ):
        return _to_jsonable(
            vars(value)
        )

    return value

CHUNKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chunks.json"
)


def _resolve_project_path(
    path_value: str,
) -> Path:
    """
    将配置中的相对路径转换为
    项目根目录下的绝对路径。
    """

    path = Path(
        path_value
    )

    if path.is_absolute():
        return path

    return (
        PROJECT_ROOT
        / path
    )


def _load_rag_chunks() -> list[dict[str, Any]]:
    """
    加载已经重新切分完成的
    RAG chunks。

    对应：
    data/processed/chunks.json
    """

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            "找不到RAG切分文件："
            f"{CHUNKS_PATH}"
        )

    with CHUNKS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(
            file
        )

    if isinstance(
        data,
        list,
    ):
        chunks = data

    elif isinstance(
        data,
        dict,
    ):
        chunks = (
            data.get("chunks")
            or data.get("documents")
            or data.get("data")
        )

    else:
        raise ValueError(
            "chunks.json必须是"
            "列表或字典"
        )

    if not chunks:
        raise ValueError(
            "chunks.json中没有"
            "有效Chunk"
        )

    return chunks


# =============================
# 初始化真实 RAG
# =============================

settings = get_settings()


embedding_model = EmbeddingModel(
    api_key=(
        settings.embedding_api_key
    ),
    base_url=(
        settings.embedding_base_url
    ),
    model=(
        settings.embedding_model
    ),
    dimensions=(
        settings.embedding_dimensions
    ),
)


vector_store = VectorStore(
    persist_directory=(
        _resolve_project_path(
            settings.chroma_dir
        )
    ),
    collection_name=(
        settings.chroma_collection
    ),
)


rag_chunks = (
    _load_rag_chunks()
)


bm25_retriever = BM25Retriever(
    rag_chunks
)


hybrid_retriever = (
    HybridRetriever(
        embedding_model=(
            embedding_model
        ),
        vector_store=(
            vector_store
        ),
        bm25_retriever=(
            bm25_retriever
        ),
    )
)


def rag_search_fn(
    question: str,
    top_k: int,
) -> list[Any]:
    """
    MCP Server真正使用的
    Hybrid RAG检索函数。

    Dense
    +
    BM25
    +
    Hybrid融合
    """

    query_embedding = (
        embedding_model.embed_query(
            question
        )
    )

    candidate_k = max(
        10,
        top_k * 2,
    )

    results = (
        hybrid_retriever.search(
            question=question,
            top_k=top_k,
            candidate_k=(
                candidate_k
            ),
            query_embedding=(
                query_embedding
            ),
        )
    )

    return results


# =============================
# Tool Registry
# =============================

registry = (
    create_signal_lab_tool_registry(
        search_fn=rag_search_fn,
    )
)

mcp = MCPServer(
    "SignalLab Metrics Server",
    instructions=(
        "用于查询和比较SignalLab项目中的真实实验指标。"
        "不得混淆不同指标、分组和实验协议。"
        "未知数据应返回明确错误，不得自行编造数值。"
    ),
)


def execute_registry_tool(
    name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """
    调用项目现有ToolRegistry，并转换成MCP返回结构。
    """
    result = registry.execute(
        name=name,
        arguments=arguments,
    )

    if not result.ok:
        error_payload = {
            "ok": False,
            "tool_name": result.tool_name,
            "error_type": result.error_type,
            "error_message": result.error_message,
        }

        raise ToolError(
            json.dumps(
                error_payload,
                ensure_ascii=False,
            )
        )

    return {
        "ok": True,
        "tool_name": result.tool_name,
        "data": result.output,
    }


@mcp.tool(
    title="查询模型实验指标",
)
def get_model_metrics(
    model: str,
    metric: str,
    group: str,
) -> dict[str, Any]:
    """
    查询一个模型在指定分组中的真实实验指标。

    Args:
        model:
            模型名称，例如NCSD-Net、TFA-Net。
        metric:
            指标名称，例如hit_rate、mse、peak_f1。
        group:
            实验分组，例如overall、low_snr。

    Returns:
        包含指标值、实验信息和来源的结构化结果。
    """
    return execute_registry_tool(
        name="get_model_metrics",
        arguments={
            "model": model,
            "metric": metric,
            "group": group,
        },
    )


@mcp.tool(
    title="比较两个模型的实验指标",
)
def compare_models(
    model_a: str,
    model_b: str,
    metric: str,
    group: str,
) -> dict[str, Any]:
    """
    比较两个模型在相同分组、相同指标下的实验结果。

    Args:
        model_a:
            第一个模型名称。
        model_b:
            第二个模型名称。
        metric:
            要比较的指标名称。
        group:
            要比较的实验分组。

    Returns:
        两个模型的指标值、比较方向和结构化结果。
    """
    return execute_registry_tool(
        name="compare_models",
        arguments={
            "model_a": model_a,
            "model_b": model_b,
            "metric": metric,
            "group": group,
        },
    )

@mcp.tool()
def search_research_docs(
    question: str,
    top_k: int = 3,
) -> str:
    """
    从研究知识库中检索相关资料。

    Args:
        question:
            用户的研究问题。

        top_k:
            返回的相关文档数量，
            默认为3。
    """

    try:
        result = registry.execute(
            "search_research_docs",
            {
                "question": question,
                "top_k": top_k,
            },
        )

        return json.dumps(
            _to_jsonable(result),
            ensure_ascii=False,
        )

    except Exception as error:
        raise ToolError(
            json.dumps(
                {
                    "ok": False,
                    "tool_name": (
                        "search_research_docs"
                    ),
                    "error_type": (
                        "tool_execution_error"
                    ),
                    "error_message": (
                        f"{type(error).__name__}: "
                        f"{error}"
                    ),
                },
                ensure_ascii=False,
            )
        ) from error

@mcp.tool()
def search_research_docs_multi(
    questions: list[str],
    top_k: int = 8,
    per_query_k: int = 6,
) -> str:
    """
    多 Query 复杂研究知识检索。
    """

    try:
        result = registry.execute(
            "search_research_docs_multi",
            {
                "questions": (
                    questions
                ),
                "top_k": top_k,
                "per_query_k": (
                    per_query_k
                ),
            },
        )

        return json.dumps(
            _to_jsonable(
                result
            ),
            ensure_ascii=False,
        )

    except Exception as error:
        raise ToolError(
            json.dumps(
                {
                    "ok": False,
                    "tool_name": (
                        "search_research_docs_multi"
                    ),
                    "error_type": (
                        "tool_execution_error"
                    ),
                    "error_message": (
                        f"{type(error).__name__}: "
                        f"{error}"
                    ),
                },
                ensure_ascii=False,
            )
        ) from error

@mcp.tool()
def list_model_metrics(
    model: str,
    group: str,
) -> str:
    try:
        result = registry.execute(
            "list_model_metrics",
            {
                "model": model,
                "group": group,
            },
        )

        return json.dumps(
            _to_jsonable(result),
            ensure_ascii=False,
        )

    except Exception as error:
        raise ToolError(
            json.dumps(
                {
                    "ok": False,
                    "tool_name": (
                        "list_model_metrics"
                    ),
                    "error_type": (
                        "tool_execution_error"
                    ),
                    "error_message": (
                        f"{type(error).__name__}: "
                        f"{error}"
                    ),
                },
                ensure_ascii=False,
            )
        ) from error


@mcp.tool()
def analyze_model_performance(
    models: list[str],
    group: str,
    metrics: list[str] | None = None,
    exclude_metrics: list[str] | None = None,
) -> str:
    try:
        result = registry.execute(
            "analyze_model_performance",
            {
                "models": models,
                "group": group,
                "metrics": metrics,
                "exclude_metrics": (
                    exclude_metrics
                ),
            },
        )

        return json.dumps(
            _to_jsonable(result),
            ensure_ascii=False,
        )

    except Exception as error:
        raise ToolError(
            json.dumps(
                {
                    "ok": False,
                    "tool_name": (
                        "analyze_model_performance"
                    ),
                    "error_type": (
                        "tool_execution_error"
                    ),
                    "error_message": (
                        f"{type(error).__name__}: "
                        f"{error}"
                    ),
                },
                ensure_ascii=False,
            )
        ) from error


@mcp.tool()
def get_metric_definition(
    metric: str,
) -> str:
    try:
        result = registry.execute(
            "get_metric_definition",
            {
                "metric": metric,
            },
        )

        return json.dumps(
            _to_jsonable(result),
            ensure_ascii=False,
        )

    except Exception as error:
        raise ToolError(
            json.dumps(
                {
                    "ok": False,
                    "tool_name": (
                        "get_metric_definition"
                    ),
                    "error_type": (
                        "tool_execution_error"
                    ),
                    "error_message": (
                        f"{type(error).__name__}: "
                        f"{error}"
                    ),
                },
                ensure_ascii=False,
            )
        ) from error


@mcp.tool()
def get_protocol_info(
    metric: str,
    group: str,
) -> str:
    try:
        result = registry.execute(
            "get_protocol_info",
            {
                "metric": metric,
                "group": group,
            },
        )

        return json.dumps(
            _to_jsonable(result),
            ensure_ascii=False,
        )

    except Exception as error:
        raise ToolError(
            json.dumps(
                {
                    "ok": False,
                    "tool_name": (
                        "get_protocol_info"
                    ),
                    "error_type": (
                        "tool_execution_error"
                    ),
                    "error_message": (
                        f"{type(error).__name__}: "
                        f"{error}"
                    ),
                },
                ensure_ascii=False,
            )
        ) from error

if __name__ == "__main__":
    mcp.run()