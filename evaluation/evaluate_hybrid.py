import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from app.rag.bm25_retriever import BM25Retriever
from app.rag.embedding_model import EmbeddingModel
from app.rag.hybrid_retriever import HybridRetriever
from app.rag.vector_store import VectorStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "retrieval_dataset.json"
)

CHUNKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chunks.json"
)


def load_dataset() -> list[dict]:
    with DATASET_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        dataset = json.load(file)

    if not isinstance(dataset, list):
        raise ValueError(
            "retrieval_dataset.json最外层必须是数组"
        )

    if not dataset:
        raise ValueError(
            "检索测试集不能为空"
        )

    return dataset


def load_chunks() -> list[dict]:
    """
    注意：

    真实文件是：
    data/processed/chunks.json

    不是：
    data/processed_chunks.json
    """

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"找不到切分文件：{CHUNKS_PATH}"
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
        raise ValueError(
            "chunks.json必须是列表或字典"
        )

    if not chunks:
        raise ValueError(
            "chunks.json中没有有效Chunk"
        )

    print(f"已加载chunks数量：{len(chunks)}")

    return chunks


def get_chunk_id(item: Any) -> str:
    if isinstance(item, dict):
        chunk_id = item.get("chunk_id")

        if chunk_id is None:
            chunk_id = item.get("id")

    else:
        chunk_id = getattr(
            item,
            "chunk_id",
            None,
        )

        if chunk_id is None:
            chunk_id = getattr(
                item,
                "id",
                None,
            )

    if chunk_id is None:
        raise ValueError(
            f"检索结果缺少chunk_id：{item}"
        )

    return str(chunk_id)


def find_first_relevant_rank(
    retrieved_chunk_ids: list[str],
    expected_chunk_ids: set[str],
) -> int | None:
    for rank, chunk_id in enumerate(
        retrieved_chunk_ids,
        start=1,
    ):
        if chunk_id in expected_chunk_ids:
            return rank

    return None


def calculate_recall_at_k(
    retrieved_chunk_ids: list[str],
    expected_chunk_ids: set[str],
    k: int,
) -> float:
    retrieved_at_k = set(
        retrieved_chunk_ids[:k]
    )

    correct_count = len(
        retrieved_at_k.intersection(
            expected_chunk_ids
        )
    )

    return correct_count / len(
        expected_chunk_ids
    )


def resolve_project_path(
    path_value: str,
) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def main() -> None:
    load_dotenv(
        PROJECT_ROOT / ".env"
    )

    dataset = load_dataset()
    chunks = load_chunks()

    embedding_model = EmbeddingModel(
        api_key=os.getenv(
            "EMBEDDING_API_KEY"
        ),
        base_url=os.getenv(
            "EMBEDDING_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
        model=os.getenv(
            "EMBEDDING_MODEL",
            "qwen3.7-text-embedding",
        ),
    )

    chroma_dir = resolve_project_path(
        os.getenv(
            "CHROMA_DIR",
            "data/chroma",
        )
    )

    chroma_collection = os.getenv(
        "CHROMA_COLLECTION",
        "signal_knowledge",
    )

    vector_store = VectorStore(
        persist_directory=chroma_dir,
        collection_name=chroma_collection,
    )

    bm25_retriever = BM25Retriever(
        chunks
    )

    hybrid_retriever = HybridRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
        bm25_retriever=bm25_retriever,
    )

    methods = (
        "Dense",
        "BM25",
        "Hybrid",
    )

    ks = (1, 3, 5)

    recall_sums = {
        method: {
            k: 0.0
            for k in ks
        }
        for method in methods
    }

    reciprocal_rank_sums = {
        method: 0.0
        for method in methods
    }

    case_analysis = []

    print("=" * 80)
    print("Dense / BM25 / Hybrid检索评测")
    print(f"测试问题数量：{len(dataset)}")
    print("=" * 80)

    for case in dataset:
        question = case["question"]

        expected_chunk_ids = set(
            case["expected_chunk_ids"]
        )

        # 每个问题只调用一次Embedding API
        query_embedding = (
            embedding_model.embed_query(
                question
            )
        )

        # Dense检索
        dense_results = vector_store.search(
            query_embedding=query_embedding,
            top_k=max(ks),
        )

        dense_chunk_ids = [
            get_chunk_id(result)
            for result in dense_results
        ]

        # BM25检索
        bm25_results = bm25_retriever.search(
            question,
            top_k=max(ks),
        )

        bm25_chunk_ids = [
            get_chunk_id(result)
            for result in bm25_results
        ]

        # Hybrid检索
        hybrid_results = hybrid_retriever.search(
            question=question,
            top_k=max(ks),
            candidate_k=10,
            query_embedding=query_embedding,
        )

        hybrid_chunk_ids = [
            get_chunk_id(result)
            for result in hybrid_results
        ]

        result_map = {
            "Dense": dense_chunk_ids,
            "BM25": bm25_chunk_ids,
            "Hybrid": hybrid_chunk_ids,
        }

        first_ranks = {}
        recall_at_3 = {}

        for method in methods:
            retrieved_chunk_ids = result_map[method]

            first_rank = find_first_relevant_rank(
                retrieved_chunk_ids,
                expected_chunk_ids,
            )

            first_ranks[method] = first_rank

            for k in ks:
                recall_sums[method][k] += (
                    calculate_recall_at_k(
                        retrieved_chunk_ids,
                        expected_chunk_ids,
                        k,
                    )
                )

            if first_rank is not None:
                reciprocal_rank_sums[method] += (
                    1 / first_rank
                )

            recall_at_3[method] = (
                calculate_recall_at_k(
                    retrieved_chunk_ids,
                    expected_chunk_ids,
                    3,
                )
            )

        best_recall_at_3 = max(
            recall_at_3.values()
        )

        best_methods = [
            method
            for method, value
            in recall_at_3.items()
            if value == best_recall_at_3
        ]

        if "Hybrid" in best_methods:
            observation = (
                "Hybrid在本题Recall@3达到最高或并列最高"
            )
        elif recall_at_3["Hybrid"] < best_recall_at_3:
            observation = (
                "Hybrid本题没有超过最优单路检索，"
                "需要检查两个检索器的候选结果"
            )
        else:
            observation = (
                "三种方法本题Recall@3相同，"
                "需要继续比较排序位置"
            )

        case_analysis.append(
            {
                "id": case["id"],
                "question": question,
                "expected_chunk_ids": sorted(
                    expected_chunk_ids
                ),
                "top3": {
                    method: result_map[method][:3]
                    for method in methods
                },
                "recall_at_3": recall_at_3,
                "first_relevant_rank": first_ranks,
                "best_methods": best_methods,
                "observation": observation,
            }
        )

        print()
        print(f"[{case['id']}] {question}")
        print(
            "正确Chunk：",
            sorted(expected_chunk_ids),
        )

        for method in methods:
            print(
                f"{method:<8}：",
                result_map[method],
                f"| Recall@3="
                f"{recall_at_3[method]:.4f}",
                f"| FirstRank="
                f"{first_ranks[method]}",
            )

    total = len(dataset)

    print()
    print("=" * 80)
    print("三种检索方法最终评测结果")
    print("=" * 80)
    print(
        f"{'指标':<15}"
        f"{'Dense':<15}"
        f"{'BM25':<15}"
        f"{'Hybrid':<15}"
    )
    print("-" * 60)

    for k in ks:
        print(
            f"{f'Recall@{k}':<15}"
            f"{recall_sums['Dense'][k] / total:<15.4f}"
            f"{recall_sums['BM25'][k] / total:<15.4f}"
            f"{recall_sums['Hybrid'][k] / total:<15.4f}"
        )

    print(
        f"{'MRR@5':<15}"
        f"{reciprocal_rank_sums['Dense'] / total:<15.4f}"
        f"{reciprocal_rank_sums['BM25'] / total:<15.4f}"
        f"{reciprocal_rank_sums['Hybrid'] / total:<15.4f}"
    )

    output_path = (
        PROJECT_ROOT
        / "evaluation"
        / "retrieval_case_analysis_day15.json"
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