from app.rag.bm25_retriever import BM25Retriever


def test_bm25_returns_relevant_chunk():
    chunks = [
        {
            "chunk_id": "chunk_001",
            "content": "多窗口FFT可以从多个分析窗口获得频谱信息。",
        },
        {
            "chunk_id": "chunk_002",
            "content": "hit rate用于衡量谱峰匹配情况。",
        },
        {
            "chunk_id": "chunk_003",
            "content": "weighted l2用于衡量频谱误差。",
        },
    ]

    retriever = BM25Retriever(chunks)
    results = retriever.search("多窗口FFT有什么作用？", top_k=3)

    assert len(results) == 3
    assert results[0]["chunk_id"] == "chunk_001"