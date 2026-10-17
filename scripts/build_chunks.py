import json
from pathlib import Path

from app.rag.document_loader import load_documents
from app.rag.text_splitter import split_document


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "chunks.json"


def main() -> None:
    documents = load_documents(RAW_DATA_DIR)

    all_chunks = []

    for document in documents:
        chunks = split_document(
            document=document,
            chunk_size=200,
            chunk_overlap=40,
        )
        all_chunks.extend(chunks)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    output_data = [
        chunk.model_dump()
        for chunk in all_chunks
    ]

    OUTPUT_FILE.write_text(
        json.dumps(
            output_data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"读取文档数量：{len(documents)}")
    print(f"生成Chunk数量：{len(all_chunks)}")
    print(f"保存位置：{OUTPUT_FILE}")


if __name__ == "__main__":
    main()