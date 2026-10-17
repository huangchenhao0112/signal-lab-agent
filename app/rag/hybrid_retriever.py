from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class HybridSearchResult:
    """Hybrid Search最终返回的结果。"""

    chunk_id: str
    content: str
    metadata: dict[str, Any]
    rrf_score: float
    dense_rank: int | None
    bm25_rank: int | None


class HybridRetriever:
    """
    使用Dense Retrieval和BM25进行混合检索。

    Dense部分使用已有的VectorStore。
    BM25部分使用已有的BM25Retriever。
    最后使用RRF融合两个检索结果。
    """

    def __init__(
        self,
        embedding_model: Any,
        vector_store: Any,
        bm25_retriever: Any,
        rank_constant: int = 60,
    ) -> None:
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.rank_constant = rank_constant

    @staticmethod
    def _read_value(
        item: Any,
        field_name: str,
        default: Any = None,
    ) -> Any:
        """
        同时兼容字典结果和对象结果。

        Dense结果通常是对象：
            result.chunk_id

        BM25结果通常是字典：
            result["chunk_id"]
        """
        if isinstance(item, dict):
            return item.get(field_name, default)

        return getattr(item, field_name, default)

    @classmethod
    def _get_chunk_id(cls, item: Any) -> str | None:
        chunk_id = cls._read_value(item, "chunk_id")

        if chunk_id is None:
            chunk_id = cls._read_value(item, "id")

        if chunk_id is None:
            return None

        return str(chunk_id)

    @classmethod
    def _get_content(cls, item: Any) -> str:
        for field_name in (
            "content",
            "text",
            "page_content",
        ):
            value = cls._read_value(item, field_name)

            if value:
                return str(value)

        return ""

    @classmethod
    def _get_metadata(
        cls,
        item: Any,
    ) -> dict[str, Any]:
        metadata = cls._read_value(
            item,
            "metadata",
            {},
        )

        if isinstance(metadata, dict):
            return metadata

        return {}

    def search(
        self,
        question: str,
        top_k: int = 5,
        candidate_k: int = 10,
        query_embedding: list[float] | None = None,
    ) -> list[HybridSearchResult]:
        """
        执行混合检索。

        参数：
            question:
                用户问题。

            top_k:
                最终返回多少条结果。

            candidate_k:
                Dense和BM25各自先召回多少条候选结果。

            query_embedding:
                可选的查询向量。
                如果外部已经生成过，就直接传进来，
                避免重复调用Embedding API。
        """

        if query_embedding is None:
            query_embedding = self.embedding_model.embed_query(
                question
            )

        dense_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=candidate_k,
        )

        bm25_results = self.bm25_retriever.search(
            question,
            top_k=candidate_k,
        )

        fused: dict[str, dict[str, Any]] = {}

        # 处理Dense结果
        for rank, result in enumerate(
            dense_results,
            start=1,
        ):
            chunk_id = self._get_chunk_id(result)

            if chunk_id is None:
                continue

            row = fused.setdefault(
                chunk_id,
                {
                    "chunk_id": chunk_id,
                    "content": "",
                    "metadata": {},
                    "rrf_score": 0.0,
                    "dense_rank": None,
                    "bm25_rank": None,
                },
            )

            row["rrf_score"] += 1 / (
                self.rank_constant + rank
            )
            row["dense_rank"] = rank

            content = self._get_content(result)

            if content and not row["content"]:
                row["content"] = content

            metadata = self._get_metadata(result)

            if metadata and not row["metadata"]:
                row["metadata"] = metadata

        # 处理BM25结果
        for rank, result in enumerate(
            bm25_results,
            start=1,
        ):
            chunk_id = self._get_chunk_id(result)

            if chunk_id is None:
                continue

            row = fused.setdefault(
                chunk_id,
                {
                    "chunk_id": chunk_id,
                    "content": "",
                    "metadata": {},
                    "rrf_score": 0.0,
                    "dense_rank": None,
                    "bm25_rank": None,
                },
            )

            row["rrf_score"] += 1 / (
                self.rank_constant + rank
            )
            row["bm25_rank"] = rank

            content = self._get_content(result)

            if content and not row["content"]:
                row["content"] = content

            metadata = self._get_metadata(result)

            if metadata and not row["metadata"]:
                row["metadata"] = metadata

        # 根据RRF分数从高到低排序
        sorted_rows = sorted(
            fused.values(),
            key=lambda row: (
                -row["rrf_score"],
                row["chunk_id"],
            ),
        )

        return [
            HybridSearchResult(
                chunk_id=row["chunk_id"],
                content=row["content"],
                metadata=row["metadata"],
                rrf_score=row["rrf_score"],
                dense_rank=row["dense_rank"],
                bm25_rank=row["bm25_rank"],
            )
            for row in sorted_rows[:top_k]
        ]