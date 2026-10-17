from app.rag.schemas import DocumentChunk
from app.rag.vector_store import VectorStore


def test_vector_store_search(tmp_path):
    vector_store = VectorStore(
        persist_directory=tmp_path / "chroma",
        collection_name="test_collection",
    )

    chunks = [
        DocumentChunk(
            chunk_id="chunk_fft",
            source="fft.md",
            chunk_index=0,
            content="FFT可以分析信号的频率成分。",
        ),
        DocumentChunk(
            chunk_id="chunk_noise",
            source="noise.md",
            chunk_index=0,
            content="低信噪比会影响弱信号检测。",
        ),
    ]

    embeddings = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]

    vector_store.upsert_chunks(
        chunks=chunks,
        embeddings=embeddings,
    )

    results = vector_store.search(
        query_embedding=[1.0, 0.0],
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].chunk_id == "chunk_fft"
    assert results[0].source == "fft.md"