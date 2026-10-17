from __future__ import annotations

import re
from typing import Any

import jieba
from rank_bm25 import BM25Okapi


class BM25Retriever:
    """
    基于 BM25 的稀疏检索器。

    输入：
        chunks: 已经切分好的文本块列表

    每个 chunk 可以是：
        {
            "chunk_id": "xxx",
            "content": "文本内容"
        }

    或者具有 chunk_id/content 属性的对象。
    """

    def __init__(self, chunks: list[Any]):
        if not chunks:
            raise ValueError("chunks 不能为空")

        self.chunks = chunks

        tokenized_corpus = [
            self._tokenize(self._get_content(chunk))
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(tokenized_corpus)

    @staticmethod
    def _get_content(chunk: Any) -> str:
        """兼容字典和对象两种 chunk 格式。"""

        if isinstance(chunk, dict):
            content = (
                chunk.get("content")
                or chunk.get("text")
                or chunk.get("page_content")
            )
        else:
            content = (
                getattr(chunk, "content", None)
                or getattr(chunk, "text", None)
                or getattr(chunk, "page_content", None)
            )

        if not content:
            raise ValueError(f"chunk缺少文本内容: {chunk}")

        return str(content)

    @staticmethod
    def _get_chunk_id(chunk: Any) -> str:
        """兼容字典和对象两种 chunk 格式。"""

        if isinstance(chunk, dict):
            chunk_id = (
                chunk.get("chunk_id")
                or chunk.get("id")
                or chunk.get("document_id")
            )
        else:
            chunk_id = (
                getattr(chunk, "chunk_id", None)
                or getattr(chunk, "id", None)
                or getattr(chunk, "document_id", None)
            )

        if chunk_id is None:
            raise ValueError(f"chunk缺少ID: {chunk}")

        return str(chunk_id)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """
        中文分词。
        英文、数字和中文词语都保留下来。
        """

        text = re.sub(r"\s+", " ", text).strip()

        tokens = []
        for word in jieba.lcut(text):
            word = word.strip()

            if not word:
                continue

            # 去掉大部分标点符号
            if re.fullmatch(r"[\W_]+", word, flags=re.UNICODE):
                continue

            tokens.append(word.lower())

        return tokens

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        返回 BM25 得分最高的 Top-K 结果。
        """

        if not query or not query.strip():
            return []

        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")

        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        results = []

        for index in ranked_indices[:top_k]:
            chunk = self.chunks[index]

            results.append(
                {
                    "chunk_id": self._get_chunk_id(chunk),
                    "content": self._get_content(chunk),
                    "score": float(scores[index]),
                }
            )

        return results