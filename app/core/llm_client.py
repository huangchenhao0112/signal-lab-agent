from dataclasses import dataclass
from typing import Optional

import httpx

from app.core.config import Settings


@dataclass
class LLMResult:
    """
    统一封装LLM调用结果。

    ok=True表示成功；
    ok=False表示调用失败。
    """

    ok: bool
    content: Optional[str] = None
    model: Optional[str] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    status_code: Optional[int] = None


class LLMClient:
    """
    项目统一的大模型调用客户端。

    当前支持OpenAI兼容的/chat/completions接口。
    """

    def __init__(self, settings: Settings):
        self.api_key = settings.llm_api_key
        self.base_url = settings.llm_base_url.rstrip("/")
        self.default_model = settings.llm_model

        self.timeout = httpx.Timeout(
            connect=10.0,
            read=settings.llm_timeout,
            write=20.0,
            pool=10.0,
        )

        self.http_client = httpx.Client(
            timeout=self.timeout,
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.2,
    ) -> LLMResult:
        """
        调用大模型聊天接口。

        messages示例：
        [
            {"role": "system", "content": "你是一名助手"},
            {"role": "user", "content": "请解释FFT"}
        ]
        """

        selected_model = model or self.default_model

        request_body = {
            "model": selected_model,
            "messages": messages,
            "temperature": temperature,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = self.http_client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=request_body,
            )

            response.raise_for_status()

            response_data = response.json()

            content = (
                response_data["choices"][0]
                ["message"]["content"]
            )

            return LLMResult(
                ok=True,
                content=content,
                model=selected_model,
                status_code=response.status_code,
            )

        except httpx.TimeoutException:
            return LLMResult(
                ok=False,
                model=selected_model,
                error_type="timeout",
                error_message="大模型请求超时",
            )

        except httpx.HTTPStatusError as error:
            return LLMResult(
                ok=False,
                model=selected_model,
                error_type="api_error",
                error_message=error.response.text[:500],
                status_code=error.response.status_code,
            )

        except httpx.RequestError as error:
            return LLMResult(
                ok=False,
                model=selected_model,
                error_type="network_error",
                error_message=str(error),
            )

        except (KeyError, IndexError, TypeError, ValueError) as error:
            return LLMResult(
                ok=False,
                model=selected_model,
                error_type="response_error",
                error_message=f"模型返回格式异常：{error}",
            )

    def close(self):
        """关闭HTTP客户端连接。"""

        self.http_client.close()