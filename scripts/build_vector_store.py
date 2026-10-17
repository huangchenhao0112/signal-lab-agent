import json
from pathlib import Path

from app.core.config import get_settings
from app.rag.embedding_model import EmbeddingModel
from app.rag.schemas import DocumentChunk
from app.rag.vector_store import VectorStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNKS_FILE = PROJECT_ROOT / "data" / "processed" / "chunks.json"


def main() -> None:
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            "没有找到chunks.json，请先运行："
            "python -m scripts.build_chunks"
        )

    raw_data = json.loads(
        CHUNKS_FILE.read_text(encoding="utf-8")
    )

    chunks = [
        DocumentChunk.model_validate(item)
        for item in raw_data
    ]

    if not chunks:
        raise ValueError("chunks.json中没有可用Chunk")

    settings = get_settings()

    print(f"正在加载Embedding模型：{settings.embedding_model}")

    embedding_model = EmbeddingModel(
    api_key=settings.embedding_api_key,
    base_url=settings.embedding_base_url,
    model=settings.embedding_model,
    dimensions=settings.embedding_dimensions,
    )

    texts = [chunk.content for chunk in chunks]

    print(f"正在为{len(texts)}个Chunk生成向量……")
    embeddings = embedding_model.embed_documents(texts)

    vector_store = VectorStore(
        persist_directory=PROJECT_ROOT / settings.chroma_dir,
        collection_name=settings.chroma_collection,
    )

    vector_store.upsert_chunks(
        chunks=chunks,
        embeddings=embeddings,
    )

    print(f"向量维度：{len(embeddings[0])}")
    print(f"向量库记录数：{vector_store.collection.count()}")
    print("向量数据库构建完成")


if __name__ == "__main__":
    main()