import json
import os
from pathlib import Path

from dotenv import load_dotenv

from app.rag.embedding_model import EmbeddingModel
from app.rag.vector_store import VectorStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "evaluation" / "retrieval_dataset.json"


def load_dataset() -> list[dict]:
    """读取人工标注的检索测试集。"""
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not isinstance(dataset, list):
        raise ValueError("retrieval_dataset.json最外层必须是JSON数组")

    if not dataset:
        raise ValueError("检索测试集不能为空")

    for case in dataset:
        if "id" not in case:
            raise ValueError("测试数据缺少id")

        if "question" not in case:
            raise ValueError(f"{case['id']}缺少question")

        if "expected_chunk_ids" not in case:
            raise ValueError(
                f"{case['id']}缺少expected_chunk_ids"
            )

        if not case["expected_chunk_ids"]:
            raise ValueError(
                f"{case['id']}的expected_chunk_ids不能为空"
            )

    return dataset


def find_first_relevant_rank(
    retrieved_chunk_ids: list[str],
    expected_chunk_ids: set[str],
) -> int | None:
    """返回第一个正确Chunk的排名，排名从1开始。"""
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
    """计算单个问题的Recall@K。"""
    retrieved_at_k = set(retrieved_chunk_ids[:k])

    correct_count = len(
        retrieved_at_k.intersection(expected_chunk_ids)
    )

    return correct_count / len(expected_chunk_ids)

def load_chunks() -> list[dict]:
    chunks_path = PROJECT_ROOT / "data" / "processed_chunks.json"

    if not chunks_path.exists():
        raise FileNotFoundError(
            f"找不到切分文件：{chunks_path}"
        )

    with chunks_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    # 如果JSON顶层本身就是列表
    if isinstance(data, list):
        chunks = data

    # 如果JSON顶层是字典，从常见字段中取列表
    elif isinstance(data, dict):
        chunks = (
            data.get("chunks")
            or data.get("documents")
            or data.get("data")
        )

    else:
        raise ValueError(
            "processed_chunks.json 的格式必须是列表或字典"
        )

    if not chunks:
        raise ValueError(
            "processed_chunks.json 中没有找到有效 chunks"
        )

    print(f"已加载 chunks 数量：{len(chunks)}")

    return chunks

def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    dataset = load_dataset()

    embedding_model_name = os.getenv(
        "EMBEDDING_MODEL",
        "qwen3.7-text-embedding",
    )

    chroma_dir = os.getenv(
        "CHROMA_DIR",
        "data/chroma",
    )

    chroma_collection = os.getenv(
        "CHROMA_COLLECTION",
        "signal_knowledge",
    )

    embedding_model = EmbeddingModel(
        api_key=os.getenv("EMBEDDING_API_KEY"),
        base_url=os.getenv(
        "EMBEDDING_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ),
        model=embedding_model_name,
    )

    vector_store = VectorStore(
        persist_directory=PROJECT_ROOT / chroma_dir,
        collection_name=chroma_collection,
    )

    ks = (1, 3, 5)

    recall_sums = {
        k: 0.0
        for k in ks
    }

    reciprocal_rank_sum = 0.0

    print("=" * 70)
    print("开始执行RAG检索评测")
    print(f"测试问题数量：{len(dataset)}")
    print("=" * 70)

    for case in dataset:
        question = case["question"]

        expected_chunk_ids = set(
            case["expected_chunk_ids"]
        )

        query_embedding = embedding_model.embed_query(
            question
        )

        search_results = vector_store.search(
            query_embedding=query_embedding,
            top_k=max(ks),
        )

        retrieved_chunk_ids = [
            result.chunk_id
            for result in search_results
        ]

        first_rank = find_first_relevant_rank(
            retrieved_chunk_ids,
            expected_chunk_ids,
        )

        for k in ks:
            recall_sums[k] += calculate_recall_at_k(
                retrieved_chunk_ids,
                expected_chunk_ids,
                k,
            )

        if first_rank is not None:
            reciprocal_rank_sum += 1 / first_rank

        print()
        print(f"[{case['id']}] {question}")
        print(
            "正确Chunk："
            f"{sorted(expected_chunk_ids)}"
        )
        print(
            "检索结果："
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
                f"  Top {index}: "
                f"{result.chunk_id} | "
                f"distance={result.distance:.4f}"
            )

    total = len(dataset)

    print()
    print("=" * 70)
    print("检索评测结果")
    print("=" * 70)

    for k in ks:
        recall_at_k = recall_sums[k] / total

        print(
            f"Recall@{k}: "
            f"{recall_at_k:.4f}"
        )

    mrr_at_5 = reciprocal_rank_sum / total

    print(f"MRR@5: {mrr_at_5:.4f}")


if __name__ == "__main__":
    main()