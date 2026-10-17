#连接真实 Embedding、Chroma、BM25 和 RRF
import json
import os
from pathlib import Path
from typing import Any
from typing import Callable

from dotenv import load_dotenv

from agent.langchain_agent import (
    SignalLabLangChainAgent,
)
from agent.tool_factory import (
    create_signal_lab_tool_registry,
)
from app.rag.bm25_retriever import (
    BM25Retriever,
)
from app.rag.embedding_model import (
    EmbeddingModel,
)
from app.rag.vector_store import (
    VectorStore,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHUNKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chunks.json"
)


def _required_env(
    name: str,
) -> str:
    """读取必需环境变量。"""

    value = os.getenv(name)

    if not value:
        raise ValueError(
            f".env中缺少{name}"
        )

    return value


def _get_value(
    item: Any,
    name: str,
    default: Any = None,
) -> Any:
    """同时兼容字典和对象形式的检索结果。"""

    if isinstance(item, dict):
        return item.get(
            name,
            default,
        )

    return getattr(
        item,
        name,
        default,
    )


def load_chunks() -> list[dict[str, Any]]:
    """读取已经切分完成的真实文档块。"""

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"找不到文档切分文件：{CHUNKS_PATH}"
        )

    with CHUNKS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if isinstance(data, list):
        chunks = data

    elif isinstance(data, dict):
        chunks = (
            data.get("chunks")
            or data.get("documents")
            or data.get("data")
        )

    else:
        chunks = None

    if not isinstance(chunks, list):
        raise ValueError(
            "chunks.json顶层必须是列表，"
            "或者包含chunks/documents/data列表"
        )

    if not chunks:
        raise ValueError(
            "chunks.json中没有文档块"
        )

    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            raise ValueError(
                f"第{index}个chunk不是字典"
            )

        if not chunk.get("chunk_id"):
            raise ValueError(
                f"第{index}个chunk缺少chunk_id"
            )

        if not chunk.get("content"):
            raise ValueError(
                f"第{index}个chunk缺少content"
            )

    return chunks


def _normalise_result(
    item: Any,
    fallback_chunks: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """统一Dense和BM25的检索结果格式。"""

    chunk_id = _get_value(
        item,
        "chunk_id",
    )

    if chunk_id is None:
        chunk_id = _get_value(
            item,
            "id",
        )

    if chunk_id is None:
        return None

    chunk_id = str(chunk_id)

    fallback = fallback_chunks.get(
        chunk_id,
        {},
    )

    content = (
        _get_value(item, "content")
        or _get_value(item, "text")
        or fallback.get("content")
        or ""
    )

    metadata = _get_value(
        item,
        "metadata",
        {},
    )

    if not isinstance(metadata, dict):
        metadata = {}

    source = (
        _get_value(item, "source")
        or metadata.get("source")
        or fallback.get("source")
    )

    chunk_index = _get_value(
        item,
        "chunk_index",
    )

    if chunk_index is None:
        chunk_index = metadata.get(
            "chunk_index"
        )

    if chunk_index is None:
        chunk_index = fallback.get(
            "chunk_index"
        )

    score = None

    for score_name in (
        "relevance_score",
        "rrf_score",
        "similarity",
        "score",
    ):
        score_value = _get_value(
            item,
            score_name,
        )

        if score_value is not None:
            score = float(score_value)
            break

    distance = _get_value(
        item,
        "distance",
    )

    if distance is not None:
        distance = float(distance)

    merged_metadata = dict(
        fallback.get(
            "metadata",
            {},
        )
        if isinstance(
            fallback.get("metadata"),
            dict,
        )
        else {}
    )

    merged_metadata.update(metadata)

    if source is not None:
        merged_metadata["source"] = source

    if chunk_index is not None:
        merged_metadata[
            "chunk_index"
        ] = chunk_index

    return {
        "chunk_id": chunk_id,
        "content": str(content),
        "metadata": merged_metadata,
        "score": score,
        "distance": distance,
    }


def create_real_hybrid_search_fn() -> Callable[
    ...,
    list[dict[str, Any]],
]:
    """
    创建真实混合检索函数。

    运行时使用：
    1. 通义Embedding生成查询向量；
    2. Chroma执行Dense检索；
    3. BM25执行稀疏检索；
    4. RRF合并两种排名。
    """

    load_dotenv(
        PROJECT_ROOT / ".env"
    )

    chunks = load_chunks()

    chunks_by_id = {
        str(chunk["chunk_id"]): chunk
        for chunk in chunks
    }

    embedding_model = EmbeddingModel(
        api_key=_required_env(
            "EMBEDDING_API_KEY"
        ),
        base_url=_required_env(
            "EMBEDDING_BASE_URL"
        ),
        model=_required_env(
            "EMBEDDING_MODEL"
        ),
    )

    chroma_directory = os.getenv(
        "CHROMA_DIR",
        "data/chroma",
    )

    chroma_collection = os.getenv(
        "CHROMA_COLLECTION",
        "signal_knowledge",
    )

    vector_store = VectorStore(
        persist_directory=(
            PROJECT_ROOT
            / chroma_directory
        ),
        collection_name=chroma_collection,
    )

    bm25_retriever = BM25Retriever(
        chunks
    )

    def search(
        question: str,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        cleaned_question = question.strip()

        if not cleaned_question:
            raise ValueError(
                "检索问题不能为空"
            )

        if top_k < 1 or top_k > 10:
            raise ValueError(
                "top_k必须在1到10之间"
            )

        # 先多召回一些，再经过RRF融合保留top_k
        candidate_k = max(
            top_k * 4,
            10,
        )

        query_embedding = (
            embedding_model.embed_query(
                cleaned_question
            )
        )

        dense_raw_results = (
            vector_store.search(
                query_embedding=(
                    query_embedding
                ),
                top_k=candidate_k,
            )
        )

        bm25_raw_results = (
            bm25_retriever.search(
                cleaned_question,
                top_k=candidate_k,
            )
        )

        dense_results = []

        for item in dense_raw_results:
            normalised = _normalise_result(
                item,
                chunks_by_id,
            )

            if normalised is not None:
                dense_results.append(
                    normalised
                )

        bm25_results = []

        for item in bm25_raw_results:
            normalised = _normalise_result(
                item,
                chunks_by_id,
            )

            if normalised is not None:
                bm25_results.append(
                    normalised
                )

        # Reciprocal Rank Fusion常用平滑常数
        rrf_k = 60

        fused: dict[
            str,
            dict[str, Any],
        ] = {}

        for retrieval_name, results in (
            ("dense", dense_results),
            ("bm25", bm25_results),
        ):
            for rank, item in enumerate(
                results,
                start=1,
            ):
                chunk_id = item["chunk_id"]

                if chunk_id not in fused:
                    fused[chunk_id] = {
                        **item,
                        "rrf_score": 0.0,
                        "dense_rank": None,
                        "bm25_rank": None,
                    }

                fused[chunk_id][
                    "rrf_score"
                ] += 1.0 / (
                    rrf_k + rank
                )

                fused[chunk_id][
                    f"{retrieval_name}_rank"
                ] = rank

                # 如果Dense结果有更完整信息，
                # 用它补全已有记录
                if (
                    retrieval_name == "dense"
                    and item.get("content")
                ):
                    fused[chunk_id][
                        "content"
                    ] = item["content"]

                    fused[chunk_id][
                        "metadata"
                    ] = item["metadata"]

                    fused[chunk_id][
                        "distance"
                    ] = item["distance"]

        sorted_results = sorted(
            fused.values(),
            key=lambda item: item[
                "rrf_score"
            ],
            reverse=True,
        )

        final_results = []

        for rank, item in enumerate(
            sorted_results[:top_k],
            start=1,
        ):
            final_results.append(
                {
                    "rank": rank,
                    "chunk_id": (
                        item["chunk_id"]
                    ),
                    "content": (
                        item["content"]
                    ),
                    "metadata": (
                        item["metadata"]
                    ),
                    "rrf_score": (
                        item["rrf_score"]
                    ),
                    "distance": (
                        item["distance"]
                    ),
                    "dense_rank": (
                        item["dense_rank"]
                    ),
                    "bm25_rank": (
                        item["bm25_rank"]
                    ),
                }
            )

        return final_results

    return search


def create_real_langchain_agent(
) -> SignalLabLangChainAgent:
    """创建连接全部真实工具的LangChain Agent。"""

    search_fn = (
        create_real_hybrid_search_fn()
    )

    registry = (
        create_signal_lab_tool_registry(
            search_fn=search_fn,
        )
    )

    agent = SignalLabLangChainAgent(
        registry=registry,
    )

    return agent