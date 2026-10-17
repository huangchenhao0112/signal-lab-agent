from pathlib import Path

import chromadb

from app.rag.schemas import DocumentChunk, SearchResult


class VectorStore:
    """负责向Chroma写入向量并进行相似度检索。"""

    def __init__(
        self,
        persist_directory: str | Path,
        collection_name: str,
    ) -> None:
        self.client = chromadb.PersistentClient(
            path=str(persist_directory)
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

    def upsert_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        """新增或更新Chunk及对应向量。"""

        if len(chunks) != len(embeddings):
            raise ValueError("Chunk数量与Embedding数量不一致")

        self.collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.content for chunk in chunks],
            embeddings=embeddings,
            metadatas=[
                {
                    "source": chunk.source,
                    "chunk_index": chunk.chunk_index,
                }
                for chunk in chunks
            ],
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 3,
    ) -> list[SearchResult]:
        """根据问题向量检索最相关的Chunk。"""

        record_count = self.collection.count()

        if record_count == 0:
            return []

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, record_count),
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        search_results = []

        for chunk_id, content, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
        ):
            search_results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    source=str(metadata["source"]),
                    chunk_index=int(metadata["chunk_index"]),
                    content=content,
                    distance=float(distance),
                )
            )

        return search_results