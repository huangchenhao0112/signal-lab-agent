import asyncio

from app.rag.rag_service import RAGService
from app.rag.schemas import SearchResult


class FakeEmbeddingModel:
    def embed_query(self, query: str) -> list[float]:
        return [1.0, 0.0]


class FakeVectorStore:
    def search(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[SearchResult]:
        return [
            SearchResult(
                chunk_id="chunk_001",
                source="fft_basics.md",
                chunk_index=0,
                content="FFT可以将时域信号转换到频域。",
                distance=0.1,
            )
        ]


class FakeLLMClient:
    def __init__(self) -> None:
        self.last_messages = None

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float,
    ) -> dict:
        self.last_messages = messages

        return {
            "content": "FFT可以将时域信号转换到频域。[资料1]"
        }


def test_rag_service_answer():
    fake_llm = FakeLLMClient()

    rag_service = RAGService(
        embedding_model=FakeEmbeddingModel(),
        vector_store=FakeVectorStore(),
        llm_client=fake_llm,
    )

    answer, sources = asyncio.run(
        rag_service.answer(
            question="FFT有什么作用？",
            top_k=1,
        )
    )

    assert "频域" in answer
    assert len(sources) == 1
    assert sources[0].source == "fft_basics.md"
    assert "[资料1]" in fake_llm.last_messages[1]["content"]