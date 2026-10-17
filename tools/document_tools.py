from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import asdict, is_dataclass
from typing import Any, Callable


SearchFunction = Callable[..., list[Any]]


def _to_plain_dict(
    item: Any,
) -> dict[str, Any]:
    """
    尽可能把不同形式的检索结果
    转换成统一 dict。

    支持：
    - dict
    - dataclass
    - Pydantic
    - 普通 Python 对象
    """

    if isinstance(item, dict):
        return dict(item)

    if is_dataclass(item):
        return asdict(item)

    if hasattr(
        item,
        "model_dump",
    ):
        value = item.model_dump()

        if isinstance(value, dict):
            return value

    if hasattr(
        item,
        "__dict__",
    ):
        return dict(
            vars(item)
        )

    return {
        "content": str(item)
    }


def _pick_first(
    data: dict[str, Any],
    names: list[str],
    default: Any = None,
) -> Any:
    """
    从多个候选字段名中
    取第一个存在的值。
    """

    for name in names:
        value = data.get(name)

        if value is not None:
            return value

    return default


def _normalize_result(
    item: Any,
) -> dict[str, Any]:
    """
    将旧 Retriever 的不同结果格式
    统一为 Agent 使用的格式。

    不强依赖某一种 SearchResult 类。
    """

    data = _to_plain_dict(
        item
    )

    metadata = data.get(
        "metadata",
        {},
    )

    if not isinstance(
        metadata,
        dict,
    ):
        metadata = {}

    content = _pick_first(
        data,
        [
            "content",
            "text",
            "page_content",
            "document",
            "chunk_text",
        ],
        "",
    )

    if not content:
        content = _pick_first(
            metadata,
            [
                "content",
                "text",
            ],
            "",
        )

    source = _pick_first(
        data,
        [
            "source",
            "filename",
            "file_name",
            "document_name",
        ],
        None,
    )

    if source is None:
        source = _pick_first(
            metadata,
            [
                "source",
                "filename",
                "file_name",
                "document_name",
            ],
            "unknown",
        )

    chunk_id = _pick_first(
        data,
        [
            "chunk_id",
            "chunk",
            "chunk_index",
            "index",
        ],
        None,
    )

    if chunk_id is None:
        chunk_id = _pick_first(
            metadata,
            [
                "chunk_id",
                "chunk",
                "chunk_index",
                "index",
            ],
            None,
        )

    similarity = _pick_first(
        data,
        [
            "similarity",
            "score",
            "relevance_score",
        ],
        None,
    )

    distance = _pick_first(
        data,
        [
            "distance",
        ],
        None,
    )

    return {
        "source": str(source),
        "chunk_id": chunk_id,
        "content": str(content),
        "similarity": similarity,
        "distance": distance,
        "metadata": metadata,
    }


def _make_result_id(
    result: dict[str, Any],
) -> str:
    """
    构造稳定的 chunk ID，
    用于多 Query 去重。
    """

    source = str(
        result.get(
            "source",
            "",
        )
    )

    chunk_id = result.get(
        "chunk_id"
    )

    content = str(
        result.get(
            "content",
            "",
        )
    )

    if chunk_id is not None:
        return (
            f"{source}::{chunk_id}"
        )

    digest = hashlib.sha1(
        content.encode(
            "utf-8"
        )
    ).hexdigest()

    return (
        f"{source}::{digest}"
    )


def _call_search_function(
    search_fn: SearchFunction,
    question: str,
    top_k: int,
) -> list[Any]:
    """
    兼容你旧项目里不同 search_fn 参数形式。

    优先：
    search_fn(question=..., top_k=...)

    如果旧代码参数名是 query，
    自动尝试 query=...

    最后尝试位置参数。
    """

    try:
        result = search_fn(
            question=question,
            top_k=top_k,
        )

    except TypeError:
        try:
            result = search_fn(
                query=question,
                top_k=top_k,
            )

        except TypeError:
            result = search_fn(
                question,
                top_k,
            )

    if result is None:
        return []

    return list(result)


class SearchResearchDocsTool:
    """
    兼容原有单 Query RAG Tool。

    简单问题仍然可以使用。
    """

    def __init__(
        self,
        search_fn: SearchFunction,
    ):
        self.search_fn = search_fn

    def __call__(
        self,
        question: str,
        top_k: int = 3,
    ) -> dict[str, Any]:

        # -------------------------
        # 参数校验
        # -------------------------

        if not isinstance(
            question,
            str,
        ):
            raise ValueError(
                "question必须是字符串"
            )

        question = question.strip()

        if not question:
            raise ValueError(
                "question不能为空"
            )

        if not isinstance(
            top_k,
            int,
        ):
            raise ValueError(
                "top_k必须是整数"
            )

        if (
            top_k < 1
            or top_k > 10
        ):
            raise ValueError(
                "top_k必须在1到10之间"
            )

        # -------------------------
        # 执行检索
        # -------------------------

        raw_results = (
            _call_search_function(
                search_fn=self.search_fn,
                question=question,
                top_k=top_k,
            )
        )

        results = [
            _normalize_result(item)
            for item in raw_results
        ]

        return {
            "ok": True,
            "tool": (
                "search_research_docs"
            ),
            "data": {
                "question": question,
                "count": len(results),
                "results": results,
            },
        }


class SearchResearchDocsMultiTool:
    """
    多 Query RAG 检索工具。

    用于复杂研究问题。

    流程：
    1. 多个子 Query 分别检索
    2. Reciprocal Rank Fusion
    3. Chunk 去重
    4. 来源多样性控制
    5. 返回最终 Top-K
    """

    def __init__(
        self,
        search_fn: SearchFunction,
        rrf_k: int = 60,
        per_source_limit: int = 2,
    ):
        self.search_fn = search_fn

        self.rrf_k = rrf_k

        self.per_source_limit = (
            per_source_limit
        )

    def __call__(
        self,
        questions: list[str],
        top_k: int = 8,
        per_query_k: int = 6,
    ) -> dict[str, Any]:
        """
        questions:
            Router 拆出的多个检索子问题。

        top_k:
            最终返回多少 chunk。

        per_query_k:
            每个子 Query 初始召回多少。
        """

        cleaned_questions = [
            question.strip()
            for question in questions
            if (
                isinstance(
                    question,
                    str,
                )
                and question.strip()
            )
        ]

        # 去重并保留顺序
        cleaned_questions = list(
            dict.fromkeys(
                cleaned_questions
            )
        )

        if not cleaned_questions:
            raise ValueError(
                "questions不能为空"
            )

        # -------------------------
        # RRF 累积分
        # -------------------------

        fused_scores: dict[
            str,
            float,
        ] = defaultdict(float)

        result_map: dict[
            str,
            dict[str, Any],
        ] = {}

        query_hits: dict[
            str,
            list[str],
        ] = defaultdict(list)

        best_rank: dict[
            str,
            int,
        ] = {}

        # -------------------------
        # 多 Query 检索
        # -------------------------

        for question in (
            cleaned_questions
        ):
            raw_results = (
                _call_search_function(
                    search_fn=(
                        self.search_fn
                    ),
                    question=question,
                    top_k=per_query_k,
                )
            )

            for rank, raw_item in (
                enumerate(
                    raw_results,
                    start=1,
                )
            ):
                result = (
                    _normalize_result(
                        raw_item
                    )
                )

                if not (
                    result[
                        "content"
                    ].strip()
                ):
                    continue

                result_id = (
                    _make_result_id(
                        result
                    )
                )

                # RRF：
                # 不直接比较不同检索器的
                # distance / similarity 数值。
                fused_scores[
                    result_id
                ] += (
                    1.0
                    / (
                        self.rrf_k
                        + rank
                    )
                )

                if (
                    result_id
                    not in result_map
                ):
                    result_map[
                        result_id
                    ] = result

                query_hits[
                    result_id
                ].append(
                    question
                )

                old_rank = (
                    best_rank.get(
                        result_id
                    )
                )

                if (
                    old_rank is None
                    or rank < old_rank
                ):
                    best_rank[
                        result_id
                    ] = rank

        # -------------------------
        # RRF 排序
        # -------------------------

        ranked_ids = sorted(
            fused_scores,
            key=lambda item: (
                fused_scores[item]
            ),
            reverse=True,
        )

        # -------------------------
        # 来源多样性
        # -------------------------

        selected: list[
            dict[str, Any]
        ] = []

        source_counts: dict[
            str,
            int,
        ] = defaultdict(int)

        skipped_ids: list[str] = []

        for result_id in ranked_ids:
            result = dict(
                result_map[
                    result_id
                ]
            )

            source = result.get(
                "source",
                "unknown",
            )

            if (
                source_counts[
                    source
                ]
                >= self.per_source_limit
            ):
                skipped_ids.append(
                    result_id
                )
                continue

            result[
                "rrf_score"
            ] = fused_scores[
                result_id
            ]

            result[
                "matched_queries"
            ] = query_hits[
                result_id
            ]

            result[
                "best_rank"
            ] = best_rank.get(
                result_id
            )

            selected.append(
                result
            )

            source_counts[
                source
            ] += 1

            if (
                len(selected)
                >= top_k
            ):
                break

        # -------------------------
        # 如果来源限制导致不足 top_k，
        # 再按总排名补足。
        # -------------------------

        if len(selected) < top_k:
            selected_ids = {
                _make_result_id(item)
                for item in selected
            }

            for result_id in (
                ranked_ids
            ):
                if (
                    result_id
                    in selected_ids
                ):
                    continue

                result = dict(
                    result_map[
                        result_id
                    ]
                )

                result[
                    "rrf_score"
                ] = fused_scores[
                    result_id
                ]

                result[
                    "matched_queries"
                ] = query_hits[
                    result_id
                ]

                result[
                    "best_rank"
                ] = best_rank.get(
                    result_id
                )

                selected.append(
                    result
                )

                if (
                    len(selected)
                    >= top_k
                ):
                    break

        return {
            "ok": True,
            "tool": (
                "search_research_docs_multi"
            ),
            "data": {
                "questions": (
                    cleaned_questions
                ),
                "query_count": len(
                    cleaned_questions
                ),
                "candidate_count": len(
                    result_map
                ),
                "count": len(
                    selected
                ),
                "results": selected,
            },
        }