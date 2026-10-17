from openai import OpenAI


class EmbeddingModel:
    """调用通义千问Embedding API生成文本向量。"""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        dimensions: int = 1024,
    ) -> None:
        self.model = model
        self.dimensions = dimensions

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    def embed_documents(
    self,
    texts: list[str],
    ) -> list[list[float]]:
        """
        批量生成文档Embedding,
        通义Embedding API单次最多接收20条文本，
        因此这里自动按20条一批调用。
        """

        if not texts:
            return []

        batch_size = 20
        all_embeddings: list[list[float]] = []

        for start_index in range(
            0,
            len(texts),
            batch_size,
        ):
            end_index = min(
                start_index + batch_size,
                len(texts),
            )

            batch_texts = texts[
                start_index:end_index
            ]

            print(
                "正在生成Embedding："
                f"{start_index + 1}"
                f"～{end_index}"
                f" / {len(texts)}"
            )

            response = self.client.embeddings.create(
                model=self.model,
                input=batch_texts,
                dimensions=self.dimensions,
            )

            # 按API返回的原始索引排序，
            # 保证Embedding与输入文本顺序一致
            sorted_items = sorted(
                response.data,
                key=lambda item: item.index,
            )

            batch_embeddings = [
                item.embedding
                for item in sorted_items
            ]

            if len(batch_embeddings) != len(
                batch_texts
            ):
                raise RuntimeError(
                    "Embedding返回数量与输入文本数量不一致："
                    f"输入{len(batch_texts)}条，"
                    f"返回{len(batch_embeddings)}条"
                )

            all_embeddings.extend(
                batch_embeddings
            )

        if len(all_embeddings) != len(texts):
            raise RuntimeError(
                "Embedding总数量与文档数量不一致："
                f"文档{len(texts)}条，"
                f"Embedding{len(all_embeddings)}条"
            )

        return all_embeddings

        
    def embed_query(
        self,
        query: str,
    ) -> list[float]:
        """把一个用户问题转换为向量。"""

        response = self.client.embeddings.create(
            model=self.model,
            input=[query],
            dimensions=self.dimensions,
        )

        return response.data[0].embedding