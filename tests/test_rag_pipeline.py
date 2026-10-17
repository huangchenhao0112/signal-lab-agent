from types import SimpleNamespace

from app.rag.rag_pipeline import (
    RAGPipeline,
)
from app.rag.reranker import (
    RerankedResult,
)


class FakeHybridRetriever:
    def search(
        self,
        question: str,
        top_k: int,
        candidate_k: int,
    ):
        return [
            SimpleNamespace(
                chunk_id="fft_001",
                content=(
                    "多窗口FFT可以提供不同"
                    "窗函数下的频谱参考。"
                ),
                metadata={
                    "file_name": (
                        "multi_window_fft.md"
                    ),
                    "section": "基本原理",
                },
            ),
            SimpleNamespace(
                chunk_id="model_001",
                content=(
                    "模型利用FFT参考辅助"
                    "神经网络恢复谱峰结构。"
                ),
                metadata={
                    "file_name": (
                        "model_structure.md"
                    ),
                    "section": "模型结构",
                },
            ),
        ]


class FakeHighScoreReranker:
    def rerank(
        self,
        question,
        candidates,
        top_n,
    ):
        return [
            RerankedResult(
                chunk_id="fft_001",
                content=(
                    "多窗口FFT可以提供不同"
                    "窗函数下的频谱参考。"
                ),
                metadata={
                    "file_name": (
                        "multi_window_fft.md"
                    ),
                    "section": "基本原理",
                },
                rerank_score=0.91,
                original_rank=1,
            )
        ]


class FakeLowScoreReranker:
    def rerank(
        self,
        question,
        candidates,
        top_n,
    ):
        return [
            RerankedResult(
                chunk_id="other_001",
                content="无关内容",
                metadata={},
                rerank_score=0.05,
                original_rank=1,
            )
        ]


def fake_generate(prompt: str) -> str:
    assert "[来源1]" in prompt

    return (
        "多窗口FFT用于提供不同窗函数下"
        "的频谱参考。[来源1]"
    )


def test_pipeline_returns_answer_and_source():
    pipeline = RAGPipeline(
        hybrid_retriever=(
            FakeHybridRetriever()
        ),
        reranker=(
            FakeHighScoreReranker()
        ),
        generate_fn=fake_generate,
        min_score=0.20,
    )

    result = pipeline.answer(
        "多窗口FFT有什么作用？"
    )

    assert result.can_answer is True
    assert len(result.sources) == 1
    assert "[来源1]" in result.answer
    assert "fft_001" in result.answer


def test_pipeline_refuses_low_score():
    generator_called = False

    def should_not_generate(
        prompt: str,
    ) -> str:
        nonlocal generator_called
        generator_called = True
        return "不应该生成"

    pipeline = RAGPipeline(
        hybrid_retriever=(
            FakeHybridRetriever()
        ),
        reranker=(
            FakeLowScoreReranker()
        ),
        generate_fn=(
            should_not_generate
        ),
        min_score=0.20,
    )

    result = pipeline.answer(
        "今天新加坡天气怎么样？"
    )

    assert result.can_answer is False
    assert result.sources == []
    assert generator_called is False
    assert "没有足够依据" in result.answer