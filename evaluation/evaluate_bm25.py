from __future__ import annotations

import json
from pathlib import Path

from app.rag.bm25_retriever import BM25Retriever
from evaluation.evaluate_dense import (
    calculate_recall_at_k,
    find_first_relevant_rank,
    load_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

KS = [1, 3, 5]


def load_chunks() -> list[dict]:
    """
    从 data/processed_chunks.json 加载已经切分好的文本块。
    """

    chunks_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "chunks.json"
    )

    if not chunks_path.exists():
        raise FileNotFoundError(
            f"找不到文件：{chunks_path}"
        )

    with chunks_path.open(
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
        raise ValueError(
            "processed_chunks.json格式不正确"
        )

    if not chunks:
        raise ValueError(
            "processed_chunks.json中没有有效chunks"
        )

    return chunks


def main() -> None:
    # 1. 加载测试问题
    dataset = load_dataset()

    # 2. 加载真实文本块
    chunks = load_chunks()

    print(f"测试问题数量：{len(dataset)}")
    print(f"文本块数量：{len(chunks)}")

    # 3. 建立BM25索引
    bm25_retriever = BM25Retriever(chunks)

    recall_sums = {
        k: 0.0
        for k in KS
    }

    reciprocal_rank_sum = 0.0

    # 4. 循环测试每个问题
    for case in dataset:
        question = case["question"]

        expected_chunk_ids = set(
            case["expected_chunk_ids"]
        )

        # 5. BM25检索
        search_results = bm25_retriever.search(
            question,
            top_k=max(KS),
        )

        retrieved_chunk_ids = [
            result["chunk_id"]
            for result in search_results
        ]

        # 6. 计算第一个正确结果排名
        first_rank = find_first_relevant_rank(
            retrieved_chunk_ids,
            expected_chunk_ids,
        )

        # 7. 计算Recall@1、Recall@3、Recall@5
        for k in KS:
            recall_sums[k] += (
                calculate_recall_at_k(
                    retrieved_chunk_ids,
                    expected_chunk_ids,
                    k,
                )
            )

        # 8. 计算MRR
        if first_rank is not None:
            reciprocal_rank_sum += (
                1 / first_rank
            )

        # 9. 打印单个问题结果
        print()
        print(f"[{case['id']}] {question}")
        print(
            "正确Chunk："
            f"{sorted(expected_chunk_ids)}"
        )
        print(
            "BM25检索结果："
            f"{retrieved_chunk_ids}"
        )
        print(
            "第一个正确结果排名："
            f"{first_rank}"
        )

        for index, result in enumerate(
            search_results,
            start=1,
        ):
            print(
                f"  BM25 Top {index}: "
                f"{result['chunk_id']} | "
                f"score={result['score']:.4f}"
            )

    # 10. 汇总结果
    total = len(dataset)

    print()
    print("=" * 70)
    print("BM25检索评测结果")
    print("=" * 70)

    for k in KS:
        recall_at_k = (
            recall_sums[k] / total
        )

        print(
            f"Recall@{k}: "
            f"{recall_at_k:.4f}"
        )

    mrr_at_5 = (
        reciprocal_rank_sum / total
    )

    print(f"MRR@5: {mrr_at_5:.4f}")


if __name__ == "__main__":
    main()