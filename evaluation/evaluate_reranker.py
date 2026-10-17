import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from app.rag.bm25_retriever import (
    BM25Retriever,
)
from app.rag.embedding_model import (
    EmbeddingModel,
)
from app.rag.hybrid_retriever import (
    HybridRetriever,
)
from app.rag.reranker import QwenReranker
from app.rag.vector_store import VectorStore

from evaluation.evaluate_hybrid import (
    calculate_recall_at_k,
    find_first_relevant_rank,
    load_chunks,
    load_dataset,
    resolve_project_path,
)


PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)


def main() -> None:
    load_dotenv(
        PROJECT_ROOT / ".env"
    )

    dataset = load_dataset()
    chunks = load_chunks()

    api_key = (
        os.getenv("RERANK_API_KEY")
        or os.getenv(
            "EMBEDDING_API_KEY"
        )
    )

    rerank_url = os.getenv(
        "RERANK_URL"
    )

    if not rerank_url:
        raise ValueError(
            "请先在.env中配置RERANK_URL"
        )

    embedding_model = EmbeddingModel(
        api_key=os.getenv(
            "EMBEDDING_API_KEY"
        ),
        base_url=os.getenv(
            "EMBEDDING_BASE_URL",
            (
                "https://dashscope.aliyuncs.com"
                "/compatible-mode/v1"
            ),
        ),
        model=os.getenv(
            "EMBEDDING_MODEL",
            "qwen3.7-text-embedding",
        ),
    )

    vector_store = VectorStore(
        persist_directory=(
            resolve_project_path(
                os.getenv(
                    "CHROMA_DIR",
                    "data/chroma",
                )
            )
        ),
        collection_name=os.getenv(
            "CHROMA_COLLECTION",
            "signal_knowledge",
        ),
    )

    bm25_retriever = BM25Retriever(
        chunks
    )

    hybrid_retriever = HybridRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
        bm25_retriever=bm25_retriever,
    )

    reranker = QwenReranker(
        api_key=api_key,
        api_url=rerank_url,
        model=os.getenv(
            "RERANK_MODEL",
            "qwen3-rerank",
        ),
    )

    hybrid_recall_1 = 0.0
    hybrid_recall_3 = 0.0
    hybrid_mrr = 0.0

    rerank_recall_1 = 0.0
    rerank_recall_3 = 0.0
    rerank_mrr = 0.0

    hybrid_latency_sum = 0.0
    rerank_latency_sum = 0.0

    case_analysis = []

    for case in dataset:
        question = case["question"]

        expected_ids = set(
            case["expected_chunk_ids"]
        )

        hybrid_start = (
            time.perf_counter()
        )

        hybrid_results = (
            hybrid_retriever.search(
                question=question,
                top_k=10,
                candidate_k=10,
            )
        )

        hybrid_latency = (
            time.perf_counter()
            - hybrid_start
        ) * 1000

        rerank_start = (
            time.perf_counter()
        )

        reranked_results = (
            reranker.rerank(
                question=question,
                candidates=hybrid_results,
                top_n=3,
            )
        )

        rerank_latency = (
            time.perf_counter()
            - rerank_start
        ) * 1000

        hybrid_ids = [
            result.chunk_id
            for result in hybrid_results
        ]

        reranked_ids = [
            result.chunk_id
            for result in reranked_results
        ]

        hybrid_first_rank = (
            find_first_relevant_rank(
                hybrid_ids[:3],
                expected_ids,
            )
        )

        rerank_first_rank = (
            find_first_relevant_rank(
                reranked_ids,
                expected_ids,
            )
        )

        hybrid_recall_1 += (
            calculate_recall_at_k(
                hybrid_ids,
                expected_ids,
                1,
            )
        )

        hybrid_recall_3 += (
            calculate_recall_at_k(
                hybrid_ids,
                expected_ids,
                3,
            )
        )

        rerank_recall_1 += (
            calculate_recall_at_k(
                reranked_ids,
                expected_ids,
                1,
            )
        )

        rerank_recall_3 += (
            calculate_recall_at_k(
                reranked_ids,
                expected_ids,
                3,
            )
        )

        if hybrid_first_rank is not None:
            hybrid_mrr += (
                1 / hybrid_first_rank
            )

        if rerank_first_rank is not None:
            rerank_mrr += (
                1 / rerank_first_rank
            )

        hybrid_latency_sum += (
            hybrid_latency
        )

        rerank_latency_sum += (
            rerank_latency
        )

        print()
        print(
            f"[{case['id']}] {question}"
        )
        print(
            "正确Chunk：",
            sorted(expected_ids),
        )
        print(
            "Hybrid Top-3：",
            hybrid_ids[:3],
        )
        print(
            "Reranker Top-3：",
            reranked_ids,
        )

        for index, result in enumerate(
            reranked_results,
            start=1,
        ):
            print(
                f"  Top {index}: "
                f"{result.chunk_id} | "
                f"score="
                f"{result.rerank_score:.4f} | "
                f"原排名="
                f"{result.original_rank}"
            )

        case_analysis.append(
            {
                "id": case["id"],
                "question": question,
                "expected_chunk_ids": (
                    sorted(expected_ids)
                ),
                "hybrid_top3": (
                    hybrid_ids[:3]
                ),
                "reranker_top3": (
                    reranked_ids
                ),
                "reranker_scores": [
                    {
                        "chunk_id": (
                            result.chunk_id
                        ),
                        "score": (
                            result.rerank_score
                        ),
                        "original_rank": (
                            result.original_rank
                        ),
                    }
                    for result
                    in reranked_results
                ],
                "hybrid_latency_ms": (
                    hybrid_latency
                ),
                "rerank_latency_ms": (
                    rerank_latency
                ),
            }
        )

    total = len(dataset)

    print()
    print("=" * 75)
    print(
        "Hybrid与Hybrid+Reranker对比"
    )
    print("=" * 75)

    print(
        f"{'指标':<20}"
        f"{'Hybrid':<20}"
        f"{'Hybrid+Reranker':<20}"
    )
    print("-" * 60)

    print(
        f"{'Recall@1':<20}"
        f"{hybrid_recall_1 / total:<20.4f}"
        f"{rerank_recall_1 / total:<20.4f}"
    )

    print(
        f"{'Recall@3':<20}"
        f"{hybrid_recall_3 / total:<20.4f}"
        f"{rerank_recall_3 / total:<20.4f}"
    )

    print(
        f"{'MRR@3':<20}"
        f"{hybrid_mrr / total:<20.4f}"
        f"{rerank_mrr / total:<20.4f}"
    )

    average_hybrid_latency = (
        hybrid_latency_sum / total
    )

    average_rerank_latency = (
        rerank_latency_sum / total
    )

    print(
        f"{'平均延迟(ms)':<20}"
        f"{average_hybrid_latency:<20.2f}"
        f"{average_hybrid_latency + average_rerank_latency:<20.2f}"
    )

    output_path = (
        PROJECT_ROOT
        / "evaluation"
        / "reranker_case_analysis.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            case_analysis,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(
        f"案例分析已保存：{output_path}"
    )


if __name__ == "__main__":
    main()