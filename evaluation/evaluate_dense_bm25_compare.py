import json
import os
from pathlib import Path

from dotenv import load_dotenv

from app.rag.embedding_model import EmbeddingModel
from app.rag.vector_store import VectorStore
from app.rag.bm25_retriever import BM25Retriever


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
    """读取人工标注的检索测试集。"""

    with DATASET_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        dataset = json.load(file)

    if not isinstance(dataset, list):
        raise ValueError(
            "retrieval_dataset.json最外层必须是JSON数组"
        )

    if not dataset:
        raise ValueError(
            "检索测试集不能为空"
        )

    for case in dataset:
        if "id" not in case:
            raise ValueError(
                "测试数据缺少id"
            )

        if "question" not in case:
            raise ValueError(
                f"{case['id']}缺少question"
            )

        if "expected_chunk_ids" not in case:
            raise ValueError(
                f"{case['id']}缺少expected_chunk_ids"
            )

        if not case["expected_chunk_ids"]:
            raise ValueError(
                f"{case['id']}的expected_chunk_ids不能为空"
            )

    return dataset


def load_chunks() -> list[dict]:
    """读取已经切分好的文本块。"""

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
            "processed_chunks.json必须是列表或字典"
        )

    if not chunks:
        raise ValueError(
            "processed_chunks.json中没有有效chunks"
        )

    return chunks


def find_first_relevant_rank(
    retrieved_chunk_ids: list[str],
    expected_chunk_ids: set[str],
) -> int | None:
    """返回第一个正确Chunk的排名。"""

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


def main() -> None:
    load_dotenv(
        PROJECT_ROOT / ".env"
    )

    dataset = load_dataset()
    chunks = load_chunks()

    print(
        f"测试问题数量：{len(dataset)}"
    )

    print(
        f"文本块数量：{len(chunks)}"
    )

    # 只初始化一次BM25
    bm25_retriever = BM25Retriever(chunks)

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
        api_key=os.getenv(
            "EMBEDDING_API_KEY"
        ),
        base_url=os.getenv(
            "EMBEDDING_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
        model=embedding_model_name,
    )

    vector_store = VectorStore(
        persist_directory=(
            PROJECT_ROOT / chroma_dir
        ),
        collection_name=chroma_collection,
    )

    ks = (1, 3, 5)

    dense_recall_sums = {
        k: 0.0
        for k in ks
    }

    bm25_recall_sums = {
        k: 0.0
        for k in ks
    }

    dense_reciprocal_rank_sum = 0.0
    bm25_reciprocal_rank_sum = 0.0
    case_analysis = []

    print("=" * 70)
    print("开始执行Dense与BM25检索对比")
    print("=" * 70)

    for case in dataset:
        question = case["question"]

        expected_chunk_ids = set(
            case["expected_chunk_ids"]
        )

        # =========================
        # Dense检索
        # =========================

        query_embedding = (
            embedding_model.embed_query(
                question
            )
        )

        dense_results = vector_store.search(
            query_embedding=query_embedding,
            top_k=max(ks),
        )

        dense_chunk_ids = [
            result.chunk_id
            for result in dense_results
        ]

        dense_first_rank = (
            find_first_relevant_rank(
                dense_chunk_ids,
                expected_chunk_ids,
            )
        )

        for k in ks:
            dense_recall_sums[k] += (
                calculate_recall_at_k(
                    dense_chunk_ids,
                    expected_chunk_ids,
                    k,
                )
            )

        if dense_first_rank is not None:
            dense_reciprocal_rank_sum += (
                1 / dense_first_rank
            )

        # =========================
        # BM25检索
        # =========================

        bm25_results = bm25_retriever.search(
            question,
            top_k=max(ks),
        )

        bm25_chunk_ids = [
            result["chunk_id"]
            for result in bm25_results
        ]

        bm25_first_rank = (
            find_first_relevant_rank(
                bm25_chunk_ids,
                expected_chunk_ids,
            )
        )

        for k in ks:
            bm25_recall_sums[k] += (
                calculate_recall_at_k(
                    bm25_chunk_ids,
                    expected_chunk_ids,
                    k,
                )
            )

        if bm25_first_rank is not None:
            bm25_reciprocal_rank_sum += (
                1 / bm25_first_rank
            )

        dense_recall_at_3 = (
            calculate_recall_at_k(
                dense_chunk_ids,
                expected_chunk_ids,
                3,
            )
        )

        bm25_recall_at_3 = (
            calculate_recall_at_k(
                bm25_chunk_ids,
                expected_chunk_ids,
                3,
            )
        )

        # Dense和BM25的Top-3结果不同，就记录为一个案例
        if (
            dense_chunk_ids[:3]
            != bm25_chunk_ids[:3]
        ):
            if (
                dense_recall_at_3
                > bm25_recall_at_3
            ):
                better_method = "Dense"
            elif (
                bm25_recall_at_3
                > dense_recall_at_3
            ):
                better_method = "BM25"
            else:
                better_method = (
                    "Recall@3相同，但排序不同"
                )

            case_analysis.append(
                {
                    "id": case["id"],
                    "question": question,
                    "expected_chunk_ids": sorted(
                        expected_chunk_ids
                    ),
                    "dense_top3": (
                        dense_chunk_ids[:3]
                    ),
                    "bm25_top3": (
                        bm25_chunk_ids[:3]
                    ),
                    "dense_recall_at_3": (
                        dense_recall_at_3
                    ),
                    "bm25_recall_at_3": (
                        bm25_recall_at_3
                    ),
                    "better_method": better_method,
                    "analysis": (
                        "需要结合两个检索结果的文本内容，"
                        "分析是关键词匹配问题还是语义匹配问题"
                    ),
                }
            )

        # =========================
        # 打印单个问题对比
        # =========================

        print()
        print(
            f"[{case['id']}] {question}"
        )

        print(
            "正确Chunk："
            f"{sorted(expected_chunk_ids)}"
        )

        print(
            "Dense结果："
            f"{dense_chunk_ids}"
        )

        print(
            "BM25结果："
            f"{bm25_chunk_ids}"
        )

        print(
            "Dense第一个正确结果排名："
            f"{dense_first_rank}"
        )

        print(
            "BM25第一个正确结果排名："
            f"{bm25_first_rank}"
        )

        print("Dense Top结果：")

        for index, result in enumerate(
            dense_results,
            start=1,
        ):
            print(
                f"  Top {index}: "
                f"{result.chunk_id} | "
                f"distance="
                f"{result.distance:.4f}"
            )

        print("BM25 Top结果：")

        for index, result in enumerate(
            bm25_results,
            start=1,
        ):
            print(
                f"  Top {index}: "
                f"{result['chunk_id']} | "
                f"score="
                f"{result['score']:.4f}"
            )

    # =========================
    # 汇总结果
    # =========================

    total = len(dataset)

    dense_mrr = (
        dense_reciprocal_rank_sum
        / total
    )

    bm25_mrr = (
        bm25_reciprocal_rank_sum
        / total
    )

    print()
    print("=" * 70)
    print("Dense与BM25检索评测对比")
    print("=" * 70)

    print(
        f"{'指标':<15}"
        f"{'Dense':<15}"
        f"{'BM25':<15}"
    )

    print("-" * 45)

    for k in ks:
        dense_recall = (
            dense_recall_sums[k]
            / total
        )

        bm25_recall = (
            bm25_recall_sums[k]
            / total
        )

        print(
            f"{'Recall@' + str(k):<15}"
            f"{dense_recall:<15.4f}"
            f"{bm25_recall:<15.4f}"
        )

    print(
        f"{'MRR@5':<15}"
        f"{dense_mrr:<15.4f}"
        f"{bm25_mrr:<15.4f}"
    )

    analysis_path = (
        PROJECT_ROOT
        / "evaluation"
        / "retrieval_case_analysis.json"
    )

    with analysis_path.open(
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
        f"已保存检索差异案例："
        f"{analysis_path}"
    )

    print()
    print("=" * 70)
    print("检索差异案例")
    print("=" * 70)

    for case_item in case_analysis:
        print(
            f"{case_item['id']} | "
            f"{case_item['question']}"
        )
        print(
            "Dense Top-3："
            f"{case_item['dense_top3']}"
        )
        print(
            "BM25 Top-3："
            f"{case_item['bm25_top3']}"
        )
        print(
            "Dense Recall@3："
            f"{case_item['dense_recall_at_3']:.4f}"
        )
        print(
            "BM25 Recall@3："
            f"{case_item['bm25_recall_at_3']:.4f}"
        )
        print(
            "结果："
            f"{case_item['better_method']}"
        )
        print()

if __name__ == "__main__":
    main()