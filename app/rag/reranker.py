from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import httpx


@dataclass
class RerankedResult:
    """重排序之后的单条结果。"""

    chunk_id: str
    content: str
    metadata: dict[str, Any]
    rerank_score: float
    original_rank: int


class RerankerError(RuntimeError):
    """Reranker调用失败。"""


class QwenReranker:
    """调用阿里云百炼Qwen Reranker。"""

    def __init__(
        self,
        api_key: str,
        api_url: str,
        model: str = "qwen3-rerank",
        timeout: float = 30.0,
    ) -> None:
        if not api_key:
            raise ValueError(
                "缺少Reranker API Key"
            )

        if not api_url:
            raise ValueError(
                "缺少RERANK_URL，请在.env中配置"
            )

        self.api_key = api_key
        self.api_url = api_url
        self.model = model
        self.timeout = timeout

    @staticmethod
    def _read_value(
        item: Any,
        field_name: str,
        default: Any = None,
    ) -> Any:
        if isinstance(item, dict):
            return item.get(field_name, default)

        return getattr(
            item,
            field_name,
            default,
        )

    @classmethod
    def _get_chunk_id(
        cls,
        item: Any,
    ) -> str:
        chunk_id = cls._read_value(
            item,
            "chunk_id",
        )

        if chunk_id is None:
            chunk_id = cls._read_value(
                item,
                "id",
            )

        if chunk_id is None:
            raise ValueError(
                "候选结果缺少chunk_id"
            )

        return str(chunk_id)

    @classmethod
    def _get_content(
        cls,
        item: Any,
    ) -> str:
        for field_name in (
            "content",
            "text",
            "page_content",
        ):
            value = cls._read_value(
                item,
                field_name,
            )

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

    def rerank(
        self,
        question: str,
        candidates: Sequence[Any],
        top_n: int = 3,
    ) -> list[RerankedResult]:
        """
        对召回结果重新排序。

        candidates可以直接接收HybridRetriever的结果。
        """

        if not question.strip():
            raise ValueError(
                "question不能为空"
            )

        if top_n <= 0:
            raise ValueError(
                "top_n必须大于0"
            )

        valid_candidates: list[
            tuple[Any, int]
        ] = []

        documents: list[str] = []

        for original_rank, candidate in enumerate(
            candidates,
            start=1,
        ):
            content = self._get_content(
                candidate
            ).strip()

            if not content:
                continue

            valid_candidates.append(
                (
                    candidate,
                    original_rank,
                )
            )
            documents.append(content)

        if not documents:
            return []

        top_n = min(
            top_n,
            len(documents),
        )

        payload = {
            "model": self.model,
            "query": question,
            "documents": documents,
            "top_n": top_n,
            "instruct": (
                "Given a research question, "
                "retrieve passages that directly "
                "provide evidence for answering it."
            ),
        }

        headers = {
            "Authorization": (
                f"Bearer {self.api_key}"
            ),
            "Content-Type": (
                "application/json"
            ),
        }

        try:
            response = httpx.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )

            response.raise_for_status()

        except httpx.TimeoutException as error:
            raise RerankerError(
                "Reranker请求超时"
            ) from error

        except httpx.HTTPStatusError as error:
            raise RerankerError(
                "Reranker接口返回错误："
                f"{error.response.status_code} "
                f"{error.response.text}"
            ) from error

        except httpx.HTTPError as error:
            raise RerankerError(
                f"Reranker网络请求失败：{error}"
            ) from error

        data = response.json()

        # qwen3-rerank的results位于顶层。
        # 这里同时兼容其他百炼Reranker格式。
        api_results = data.get("results")

        if api_results is None:
            api_results = (
                data.get("output", {})
                .get("results")
            )

        if not isinstance(
            api_results,
            list,
        ):
            raise RerankerError(
                "Reranker响应中缺少results"
            )

        reranked_results = []

        for item in api_results:
            document_index = int(
                item["index"]
            )

            if not (
                0
                <= document_index
                < len(valid_candidates)
            ):
                continue

            candidate, original_rank = (
                valid_candidates[
                    document_index
                ]
            )

            reranked_results.append(
                RerankedResult(
                    chunk_id=(
                        self._get_chunk_id(
                            candidate
                        )
                    ),
                    content=(
                        self._get_content(
                            candidate
                        )
                    ),
                    metadata=(
                        self._get_metadata(
                            candidate
                        )
                    ),
                    rerank_score=float(
                        item[
                            "relevance_score"
                        ]
                    ),
                    original_rank=(
                        original_rank
                    ),
                )
            )

        return reranked_results