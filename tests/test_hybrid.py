from types import SimpleNamespace

from app.rag.hybrid_retriever import HybridRetriever


class FakeEmbeddingModel:
    def embed_query(
        self,
        question: str,
    ) -> list[float]:
        return [0.1, 0.2, 0.3]


class FakeVectorStore:
    def search(
        self,
        query_embedding,
        top_k: int,
    ):
        return [
            SimpleNamespace(
                chunk_id="dense_only",
                content="Dense结果",
                metadata={},
            ),
            SimpleNamespace(
                chunk_id="shared",
                content="共同结果",
                metadata={},
            ),
        ]


class FakeBM25Retriever:
    def search(
        self,
        question: str,
        top_k: int,
    ):
        return [
            {
                "chunk_id": "shared",
                "content": "共同结果",
                "metadata": {},
            },
            {
                "chunk_id": "bm25_only",
                "content": "BM25结果",
                "metadata": {},
            },
        ]


def test_rrf_prefers_shared_result():
    retriever = HybridRetriever(
        embedding_model=FakeEmbeddingModel(),
        vector_store=FakeVectorStore(),
        bm25_retriever=FakeBM25Retriever(),
    )

    results = retriever.search(
        question="测试问题",
        top_k=3,
        candidate_k=5,
    )

    assert len(results) == 3

    # shared同时出现在Dense和BM25中，
    # RRF分数应该最高
    assert results[0].chunk_id == "shared"

    assert results[0].dense_rank == 2
    assert results[0].bm25_rank == 1