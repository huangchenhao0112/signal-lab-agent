import asyncio
from typing import Any

from app.rag.schemas import SearchResult


class RAGService:
    """串联Embedding、向量检索和大模型生成。"""

    def __init__(
        self,
        embedding_model,
        vector_store,
        llm_client,
    ) -> None:
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.llm_client = llm_client

    async def answer(
        self,
        question: str,
        top_k: int = 3,
    ) -> tuple[str, list[SearchResult]]:
        """根据知识库回答用户问题。"""

        # 第一步：把问题转换为向量
        query_embedding = await asyncio.to_thread(
            self.embedding_model.embed_query,
            question,
        )

        # 第二步：从Chroma检索相关Chunk
        search_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        if not search_results:
            return (
                "当前知识库中没有可用于回答该问题的资料。",
                [],
            )

        # 第三步：把检索结果整理成上下文
        context = self._build_context(search_results)

        # 第四步：构造RAG Prompt
        messages = self._build_messages(
            question=question,
            context=context,
        )

        # 第五步：调用聊天大模型生成答案
        llm_response = await asyncio.to_thread(
            self.llm_client.chat,
            messages=messages,
            temperature=0.2,
        )

        answer = self._extract_content(llm_response)

        return answer, search_results

    @staticmethod
    def _build_context(
        search_results: list[SearchResult],
    ) -> str:
        """把多个检索结果整理成带编号的上下文。"""

        context_parts = []

        for index, result in enumerate(
            search_results,
            start=1,
        ):
            context_parts.append(
                f"[资料{index}]\n"
                f"来源：{result.source}\n"
                f"内容：{result.content}"
            )

        return "\n\n".join(context_parts)

    @staticmethod
    def _build_messages(
        question: str,
        context: str,
    ) -> list[dict[str, str]]:
        """创建发送给大模型的消息。"""

        system_prompt = (
            "你是一名通信工程与信号分析知识助手。"
            "你只能根据提供的知识库资料回答问题。"
            "如果资料不足以回答，必须明确说明资料不足，"
            "不要使用未提供的信息进行猜测。"
            "回答时使用[资料1]、[资料2]等形式标注依据。"
        )

        user_prompt = (
            f"以下是从知识库中检索到的资料：\n\n"
            f"{context}\n\n"
            f"用户问题：{question}\n\n"
            "请根据上述资料回答问题，并标注所依据的资料编号。"
        )

        return [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

    @staticmethod
    def _extract_content(response: Any) -> str:
        """兼容不同LLMClient返回格式。"""

        if isinstance(response, str):
            return response

        if isinstance(response, dict):
            content = response.get("content")

            if content:
                return str(content)

        content = getattr(response, "content", None)

        if content:
            return str(content)

        raise ValueError("无法从LLM返回结果中提取content")