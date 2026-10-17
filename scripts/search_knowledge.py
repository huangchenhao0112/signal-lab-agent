from pathlib import Path

from app.core.config import get_settings
from app.rag.embedding_model import EmbeddingModel
from app.rag.vector_store import VectorStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    settings = get_settings()

    embedding_model = EmbeddingModel(
    api_key=settings.embedding_api_key,
    base_url=settings.embedding_base_url,
    model=settings.embedding_model,
    dimensions=settings.embedding_dimensions,
    )

    vector_store = VectorStore(
        persist_directory=PROJECT_ROOT / settings.chroma_dir,
        collection_name=settings.chroma_collection,
    )

    query = input("请输入检索问题：").strip()

    if not query:
        print("问题不能为空")
        return

    query_embedding = embedding_model.embed_query(query)

    results = vector_store.search(
        query_embedding=query_embedding,
        top_k=settings.top_k,
    )

    if not results:
        print("向量数据库中没有可检索内容")
        return

    print("\n检索结果：")

    for index, result in enumerate(results, start=1):
        print(f"\n--- 结果{index} ---")
        print(f"来源：{result.source}")
        print(f"Chunk：{result.chunk_index}")
        print(f"距离：{result.distance:.4f}")
        print(f"内容：{result.content}")


if __name__ == "__main__":
    main()