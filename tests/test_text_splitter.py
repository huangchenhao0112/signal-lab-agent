from pathlib import Path

from app.rag.document_loader import load_documents
from app.rag.schemas import SourceDocument
from app.rag.text_splitter import split_document


def test_split_long_document():
    document = SourceDocument(
        source="test.md",
        content="A" * 300,
    )

    chunks = split_document(
        document=document,
        chunk_size=100,
        chunk_overlap=20,
    )

    assert len(chunks) > 1
    assert all(len(chunk.content) <= 100 for chunk in chunks)
    assert chunks[0].content[-20:] == chunks[1].content[:20]


def test_chunk_metadata():
    document = SourceDocument(
        source="fft.md",
        content="FFT用于分析信号中的频率成分。" * 20,
    )

    chunks = split_document(
        document=document,
        chunk_size=100,
        chunk_overlap=20,
    )

    assert chunks[0].source == "fft.md"
    assert chunks[0].chunk_index == 0
    assert chunks[0].chunk_id != ""


def test_load_documents(tmp_path: Path):
    (tmp_path / "a.md").write_text(
        "这是Markdown文档。",
        encoding="utf-8",
    )
    (tmp_path / "b.txt").write_text(
        "这是TXT文档。",
        encoding="utf-8",
    )
    (tmp_path / "ignore.py").write_text(
        "print('这个文件不应该被读取')",
        encoding="utf-8",
    )

    documents = load_documents(tmp_path)

    assert len(documents) == 2