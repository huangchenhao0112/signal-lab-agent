from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from app.rag.reranker import (
    QwenReranker,
    RerankedResult,
)


@dataclass
class SourceReference:
    """返回给用户的来源信息。"""

    citation_id: str
    chunk_id: str
    file_name: str
    section: str
    rerank_score: float


@dataclass
class RAGAnswer:
    """RAG Pipeline最终结果。"""

    can_answer: bool
    answer: str
    sources: list[SourceReference]
    retrieved_chunks: list[RerankedResult]


class RAGPipeline:
    """
    完整流程：

    Hybrid Top-10
    → Reranker Top-3
    → 阈值过滤
    → Prompt
    → 生成回答
    → 返回引用
    """

    def __init__(
        self,
        hybrid_retriever: Any,
        reranker: QwenReranker,
        generate_fn: Callable[[str], str],
        min_score: float = 0.20,
    ) -> None:
        self.hybrid_retriever = (
            hybrid_retriever
        )
        self.reranker = reranker
        self.generate_fn = generate_fn
        self.min_score = min_score

    @staticmethod
    def _get_source_name(
        metadata: dict,
    ) -> str:
        return str(
            metadata.get("source")
            or metadata.get("file_name")
            or metadata.get("filename")
            or metadata.get("path")
            or "未知文件"
        )

    @staticmethod
    def _get_section(
        metadata: dict,
    ) -> str:
        return str(
            metadata.get("section")
            or metadata.get("title")
            or metadata.get("heading")
            or "未标注章节"
        )

    def _build_sources(
        self,
        results: list[RerankedResult],
    ) -> list[SourceReference]:
        sources = []

        for index, result in enumerate(
            results,
            start=1,
        ):
            sources.append(
                SourceReference(
                    citation_id=(
                        f"来源{index}"
                    ),
                    chunk_id=(
                        result.chunk_id
                    ),
                    file_name=(
                        self._get_source_name(
                            result.metadata
                        )
                    ),
                    section=(
                        self._get_section(
                            result.metadata
                        )
                    ),
                    rerank_score=(
                        result.rerank_score
                    ),
                )
            )

        return sources

    @staticmethod
    def _build_prompt(
        question: str,
        results: list[RerankedResult],
    ) -> str:
        context_parts = []

        for index, result in enumerate(
            results,
            start=1,
        ):
            context_parts.append(
                "\n".join(
                    [
                        (
                            f"[来源{index}] "
                            f"chunk_id="
                            f"{result.chunk_id}"
                        ),
                        result.content,
                    ]
                )
            )

        context = "\n\n".join(
            context_parts
        )

        return f"""
你是一个科研知识库助手。

请严格依据“参考资料”回答问题，不允许使用参考资料之外的信息。

要求：
1. 每个关键结论后必须标注来源，例如[来源1]。
2. 不允许编造不存在的指标、模型结构或实验结果。
3. 如果参考资料不能回答问题，必须明确回答：
   “当前知识库中没有足够依据回答该问题。”
4. 回答使用中文，保持准确、简洁。

用户问题：
{question}

参考资料：
{context}
""".strip()

    @staticmethod
    def _format_source_list(
        sources: list[SourceReference],
    ) -> str:
        lines = []

        for source in sources:
            lines.append(
                (
                    f"[{source.citation_id}] "
                    f"{source.file_name} / "
                    f"{source.section} / "
                    f"{source.chunk_id}"
                )
            )

        return "\n".join(lines)

    def answer(
        self,
        question: str,
        candidate_k: int = 10,
        final_k: int = 3,
    ) -> RAGAnswer:
        if not question.strip():
            raise ValueError(
                "question不能为空"
            )

        # 第一步：Hybrid召回Top-10
        candidates = (
            self.hybrid_retriever.search(
                question=question,
                top_k=candidate_k,
                candidate_k=candidate_k,
            )
        )

        if not candidates:
            return RAGAnswer(
                can_answer=False,
                answer=(
                    "当前知识库中没有足够依据"
                    "回答该问题。"
                ),
                sources=[],
                retrieved_chunks=[],
            )

        # 第二步：Reranker精排Top-3
        reranked_results = (
            self.reranker.rerank(
                question=question,
                candidates=candidates,
                top_n=final_k,
            )
        )

        # 第三步：过滤低于阈值的Chunk
        accepted_results = [
            result
            for result in reranked_results
            if result.rerank_score
            >= self.min_score
        ]

        # 第四步：无可靠资料，程序直接拒答
        if not accepted_results:
            return RAGAnswer(
                can_answer=False,
                answer=(
                    "当前知识库中没有足够依据"
                    "回答该问题。"
                ),
                sources=[],
                retrieved_chunks=(
                    reranked_results
                ),
            )

        # 第五步：构造带来源Prompt
        prompt = self._build_prompt(
            question=question,
            results=accepted_results,
        )

        # 第六步：调用已有LLM生成函数
        generated_answer = (
            self.generate_fn(prompt)
        ).strip()

        if not generated_answer:
            raise RuntimeError(
                "LLM返回了空答案"
            )

        sources = self._build_sources(
            accepted_results
        )

        source_list = (
            self._format_source_list(
                sources
            )
        )

        final_answer = (
            f"{generated_answer}"
            f"\n\n来源：\n"
            f"{source_list}"
        )

        return RAGAnswer(
            can_answer=True,
            answer=final_answer,
            sources=sources,
            retrieved_chunks=(
                accepted_results
            ),
        )